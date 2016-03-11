//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_EpetraMVDenseMatrixView.hpp"

#include "Epetra_MultiVector.h"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialSymDenseMatrix.h"

#include "Epetra_Comm.h"

#include "Teuchos_TestForException.hpp"

namespace MOR {

namespace { // anonymous

bool any(const Epetra_Comm &comm, bool localCondition) {
  int localValue = static_cast<int>(localCondition);
  int result;
  comm.MaxAll(&localValue, &result, 1);
  return result;
}

bool all(const Epetra_Comm &comm, bool localCondition) {
  int localValue = static_cast<int>(localCondition);
  int result;
  comm.MinAll(&localValue, &result, 1);
  return result;
}

} // end anonymous namespace

Epetra_SerialDenseMatrix localDenseMatrixView(Epetra_MultiVector &mv) {
  double *values;
  int localLeadDim;
  mv.ExtractView(&values, &localLeadDim);
  return Epetra_SerialDenseMatrix(View, values, localLeadDim, mv.MyLength(), mv.NumVectors());
}

Epetra_SerialSymDenseMatrix localSymDenseMatrixView(Epetra_MultiVector &mv) {
  const int matrixSize = mv.NumVectors();
  TEUCHOS_TEST_FOR_EXCEPT(any(mv.Comm(), matrixSize != mv.MyLength()));
  double *values;
  int localLeadDim;
  mv.ExtractView(&values, &localLeadDim);
  return Epetra_SerialSymDenseMatrix(View, values, localLeadDim, matrixSize);
}

} // namespace MOR
