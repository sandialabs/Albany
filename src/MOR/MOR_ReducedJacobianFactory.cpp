//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedJacobianFactory.hpp"

#include "MOR_BasisOps.hpp"

#include "Epetra_CrsMatrix.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Vector.h"
#include "Epetra_Comm.h"

#include "Teuchos_Ptr.hpp"
#include "Teuchos_Array.hpp"

namespace MOR {

using ::Teuchos::Ptr;
using ::Teuchos::ptr;
using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::Array;

ReducedJacobianFactory::ReducedJacobianFactory(const RCP<const Epetra_MultiVector> &rightProjector) :
  rightProjector_(rightProjector),
  premultipliedRightProjector_(new Epetra_MultiVector(*rightProjector)),
  reducedGraph_(Copy,
                Epetra_LocalMap(rightProjector->NumVectors(), 0, rightProjector->Comm()),
                Epetra_LocalMap(rightProjector->NumVectors(), 0, rightProjector->Comm()),
                rightProjector->NumVectors(),
                true /* static profile */)
{
  Teuchos::Array<int> Indices;
  for (int i=0; i<reducedGraph_.NumMyRows(); i++) {
     Indices.resize(reducedGraph_.NumMyCols());
     for (int j=0; j<reducedGraph_.NumMyCols(); j++) {
       Indices[j] = j;
     }
     int err = reducedGraph_.InsertMyIndices(i, reducedGraph_.NumMyCols(), Indices.getRawPtr());
     TEUCHOS_ASSERT(err == 0);
  }

  const int err = reducedGraph_.FillComplete();
  TEUCHOS_ASSERT(err == 0);
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

  Epetra_Vector rowVector(result.RangeMap(), false);
  for (int i = 0; i<result.NumMyRows(); i++) {
     int NumEntries; int *Indices;
     int err = reducedGraph_.ExtractMyRowView(i, NumEntries, Indices);
     TEUCHOS_ASSERT(err == 0);
     err = reduce(*premultipliedRightProjector_, *(leftProjector)(i), rowVector);
     TEUCHOS_ASSERT(err == 0);
     err = result.ReplaceMyValues(i, NumEntries, rowVector.Values(), Indices);
     TEUCHOS_ASSERT(err == 0);
  }

  return result;
}

} // namespace MOR
