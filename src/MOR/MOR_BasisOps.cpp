//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_BasisOps.hpp"

#include "MOR_EpetraMVDenseMatrixView.hpp"

#include "Epetra_BlockMap.h"
#include "Epetra_LocalMap.h"

#include "Epetra_Operator.h"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "Teuchos_Assert.hpp"

namespace MOR {

Epetra_LocalMap createComponentMap(const Epetra_MultiVector &projector)
{
  return Epetra_LocalMap(projector.NumVectors(), 0, projector.Comm());
}

void dualize(const Epetra_MultiVector &primal, Epetra_MultiVector &dual)
{
  // 1) A <- primal^T * dual
  const Epetra_LocalMap componentMap = createComponentMap(dual);
  Epetra_MultiVector product(componentMap, primal.NumVectors(), false);
  {
    const int ierr = reduce(dual, primal, product);
    TEUCHOS_ASSERT(ierr == 0);
  }

  // 2) A <- A^{-1}
  {
    Epetra_SerialDenseMatrix matrix = localDenseMatrixView(product);
    Epetra_SerialDenseSolver solver;
    {
      const int ierr = solver.SetMatrix(matrix);
      TEUCHOS_ASSERT(ierr == 0);
    }
    {
      const int ierr = solver.Invert();
      TEUCHOS_ASSERT(ierr == 0);
    }
  }

  // 3) dual <- dual * A
  const Epetra_MultiVector dual_copy(dual);
  dual.Multiply('N', 'N', 1.0, dual_copy, product, 0.0);
}

void dualize(const Epetra_MultiVector &primal, const Epetra_Operator &metric, Epetra_MultiVector &result)
{
  const int ierr = metric.Apply(primal, result);
  TEUCHOS_ASSERT(ierr == 0);
  dualize(primal, result);
}

} // namespace MOR
