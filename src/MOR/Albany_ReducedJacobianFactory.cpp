/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_ReducedJacobianFactory.hpp"

#include "Albany_BasisOps.hpp"

#include "Epetra_CrsMatrix.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Vector.h"
#include "Epetra_Comm.h"

#include "Teuchos_Ptr.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

using ::Teuchos::Ptr;
using ::Teuchos::ptr;
using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::Array;

ReducedJacobianFactory::ReducedJacobianFactory(const RCP<const Epetra_MultiVector> &rightProjector) :
  rightProjector_(rightProjector),
  premultipliedRightProjector_(new Epetra_MultiVector(*rightProjector)),
  reducedGraph_(Copy,
                Epetra_Map(rightProjector->NumVectors(), isMasterProcess() ? rightProjector->NumVectors() : 0, 0, rightProjector->Comm()),
                rightProjector->NumVectors(),
                true /* static profile */)
{
  const int rowColCount = rightProjector->NumVectors();

  if (isMasterProcess())
  {
    Array<int> entryIndices(rowColCount);
    for (int iCol = 0; iCol < entryIndices.size(); ++iCol) {
      entryIndices[iCol] = iCol;
    }

    for (int iRow = 0; iRow < rowColCount; ++iRow) {
      const int err = reducedGraph_.InsertGlobalIndices(iRow, rowColCount, entryIndices.getRawPtr());
      TEUCHOS_ASSERT(err == 0);
    }
  }

  {
    const Epetra_LocalMap replicationMap(rowColCount, 0, rightProjector->Comm());
    const int err = reducedGraph_.FillComplete(replicationMap, replicationMap);
    TEUCHOS_ASSERT(err == 0);
  }

  {
    const int err = reducedGraph_.OptimizeStorage();
    TEUCHOS_ASSERT(err == 0);
  }
}

void ReducedJacobianFactory::fullJacobianIs(const Epetra_Operator &op) {
  const int err = op.Apply(*rightProjector_, *premultipliedRightProjector_);
  TEUCHOS_ASSERT(err == 0);
}

RCP<Epetra_CrsMatrix> ReducedJacobianFactory::reducedMatrixNew() const
{
  return rcp(new Epetra_CrsMatrix(Copy, reducedGraph_));
}

const Epetra_CrsMatrix &ReducedJacobianFactory::reducedMatrix(const Epetra_MultiVector &leftProjector,
                                                              Epetra_CrsMatrix &result) const
{
  TEUCHOS_ASSERT(leftProjector.NumVectors() == rightProjector_->NumVectors());
  TEUCHOS_ASSERT(result.Filled());

  Ptr<const int> entryIndices;
  if (isMasterProcess())
  {
    int dummyEntryCount;
    int *entryIndicesTemp;
    const int err = reducedGraph_.ExtractMyRowView(0, dummyEntryCount, entryIndicesTemp);
    TEUCHOS_ASSERT(err == 0);
    entryIndices = ptr(entryIndicesTemp);
  }

  Epetra_Vector rowVector(result.RangeMap(), false);
  for (int iRow = 0; iRow < leftProjector.NumVectors(); ++iRow) {
    {
      const int err = reduce(*premultipliedRightProjector_, *(leftProjector)(iRow), rowVector);
      TEUCHOS_ASSERT(err == 0);
    }
    if (isMasterProcess())
    {
      const int err = result.ReplaceMyValues(iRow, rightProjector_->NumVectors(), rowVector.Values(), entryIndices.get());
      TEUCHOS_ASSERT(err == 0);
    }
  }

  return result;
}

bool ReducedJacobianFactory::isMasterProcess() const
{
  return rightProjector_->Comm().MyPID() == 0;
}

} // namespace Albany
