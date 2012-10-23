//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ReducedLinearOperatorFactory.hpp"

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

ReducedLinearOperatorFactory::ReducedLinearOperatorFactory(const RCP<const Epetra_MultiVector> &projector) :
  rightProjector_(projector),
  leftProjector_(projector),
  reducedGraph_(Copy,
                Epetra_Map(projector->NumVectors(), isMasterProcess() ? projector->NumVectors() : 0, 0, projector->Comm()),
                projector->NumVectors(),
                true /* static profile */)
{
  init();
}

ReducedLinearOperatorFactory::ReducedLinearOperatorFactory(const RCP<const Epetra_MultiVector> &rightProjector,
                                                           const RCP<const Epetra_MultiVector> &leftProjector) :
  rightProjector_(rightProjector),
  leftProjector_(leftProjector),
  reducedGraph_(Copy,
                Epetra_Map(leftProjector_->NumVectors(), isMasterProcess() ? leftProjector_->NumVectors() : 0, 0, leftProjector_->Comm()),
                Epetra_Map(rightProjector_->NumVectors(), isMasterProcess() ? rightProjector_->NumVectors() : 0, 0, rightProjector_->Comm()),
                rightProjector_->NumVectors(),
                true /* static profile */)
{
  init();
}

void ReducedLinearOperatorFactory::init()
{
  const int rowCount = leftProjector_->NumVectors();
  const int colCount = rightProjector_->NumVectors();

  if (isMasterProcess())
  {
    Array<int> entryIndices(colCount);
    for (int iCol = 0; iCol < entryIndices.size(); ++iCol) {
      entryIndices[iCol] = iCol;
    }

    for (int iRow = 0; iRow < rowCount; ++iRow) {
      const int err = reducedGraph_.InsertGlobalIndices(iRow, colCount, entryIndices.getRawPtr());
      TEUCHOS_ASSERT(err == 0);
    }
  }

  {
    const int err = reducedGraph_.FillComplete(Epetra_LocalMap(colCount, 0, rightProjector_->Comm()),
                                               Epetra_LocalMap(rowCount, 0, leftProjector_->Comm()));
    TEUCHOS_ASSERT(err == 0);
  }

  {
    const int err = reducedGraph_.OptimizeStorage();
    TEUCHOS_ASSERT(err == 0);
  }
}

RCP<Epetra_CrsMatrix> ReducedLinearOperatorFactory::reducedOperatorNew() const
{
  return rcp(new Epetra_CrsMatrix(Copy, reducedGraph_));
}

void ReducedLinearOperatorFactory::reducedOperatorInit(const Epetra_Operator &fullOperator,
                                                       Epetra_CrsMatrix &result) const
{
  // TODO Check arguments
  TEUCHOS_ASSERT(result.Filled());

  Epetra_MultiVector premultipliedRightProjector(fullOperator.OperatorRangeMap(), rightProjector_->NumVectors(), false);
  {
    const int err = fullOperator.Apply(*rightProjector_, premultipliedRightProjector);
    TEUCHOS_ASSERT(err == 0);
  }

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
  for (int iRow = 0; iRow < leftProjector_->NumVectors(); ++iRow) {
    {
      const int err = reduce(premultipliedRightProjector, *(*leftProjector_)(iRow), rowVector);
      TEUCHOS_ASSERT(err == 0);
    }
    if (isMasterProcess())
    {
      const int err = result.ReplaceMyValues(iRow, rightProjector_->NumVectors(), rowVector.Values(), entryIndices.get());
      TEUCHOS_ASSERT(err == 0);
    }
  }
}

RCP<Epetra_CrsMatrix> ReducedLinearOperatorFactory::reducedOperatorNew(const Epetra_Operator &fullOperator) const
{
  const RCP<Epetra_CrsMatrix> result = reducedOperatorNew();
  reducedOperatorInit(fullOperator, *result);
  return result;
}

bool ReducedLinearOperatorFactory::isMasterProcess() const
{
  return leftProjector_->Comm().MyPID() == 0;
}

} // namespace Albany
