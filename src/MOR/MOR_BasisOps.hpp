//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_BASISOPS_HPP
#define MOR_BASISOPS_HPP

#include "Epetra_MultiVector.h"

class Epetra_LocalMap;
class Epetra_Operator;

namespace MOR {

// Convenience functions for common reduced-order basis computations
Epetra_LocalMap createComponentMap(const Epetra_MultiVector &projector);

// result <- (basis * components) + (resultScaling * result)
inline
int expandAdd(const Epetra_MultiVector &basis, const Epetra_MultiVector &components, double resultScaling, Epetra_MultiVector &result)
{
  return result.Multiply('N', 'N', 1.0, basis, components, resultScaling);
}

// result <- (basis * components) + result
inline
int expandAdd(const Epetra_MultiVector &basis, const Epetra_MultiVector &components, Epetra_MultiVector &result)
{
  return expandAdd(basis, components, 1.0, result);
}

// result <- basis * components
inline
int expand(const Epetra_MultiVector &basis, const Epetra_MultiVector &components, Epetra_MultiVector &result)
{
  return expandAdd(basis, components, 0.0, result);
}

// result <- (basis^T * vectors) + (resultScaling * result)
inline
int reduceAdd(const Epetra_MultiVector &basis, const Epetra_MultiVector &vectors, double resultScaling, Epetra_MultiVector &result)
{
  return result.Multiply('T', 'N', 1.0, basis, vectors, resultScaling);
}

// result <- (basis^T * vectors) + result
inline
int reduceAdd(const Epetra_MultiVector &basis, const Epetra_MultiVector &vectors, Epetra_MultiVector &result)
{
  return reduceAdd(basis, vectors, 1.0, result);
}

// result <- basis^T * vectors
inline
int reduce(const Epetra_MultiVector &basis, const Epetra_MultiVector &vectors, Epetra_MultiVector &result)
{
  return reduceAdd(basis, vectors, 0.0, result);
}

// dual <- dual * (primal^T * dual)^{-1}
void dualize(const Epetra_MultiVector &primal, Epetra_MultiVector &dual);

// dual <- (metric * primal) * (primal^T * metric * primal)^{-1}
void dualize(const Epetra_MultiVector &primal, const Epetra_Operator &metric, Epetra_MultiVector &result);

} // namespace MOR

#endif /* MOR_BASISOPS_HPP */
