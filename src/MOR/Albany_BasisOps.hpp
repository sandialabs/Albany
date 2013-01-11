//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_BASISOPS_HPP
#define ALBANY_BASISOPS_HPP

#include "Epetra_MultiVector.h"

class Epetra_LocalMap;

namespace Albany {

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

} // end namespace Albany

#endif /* ALBANY_BASISOPS_HPP */
