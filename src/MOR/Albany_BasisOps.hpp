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

#ifndef ALBANY_BASISOPS_HPP
#define ALBANY_BASISOPS_HPP

#include "Epetra_MultiVector.h"

namespace Albany {

// Convenience functions for common reduced-order basis computations

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

// result <- basis^T * vectors
inline
int reduceAdd(const Epetra_MultiVector &basis, const Epetra_MultiVector &vectors, Epetra_MultiVector &result)
{
  return reduceAdd(basis, vectors, 1.0, result);
}

// result <- (basis^T * vectors) + result
inline
int reduce(const Epetra_MultiVector &basis, const Epetra_MultiVector &vectors, Epetra_MultiVector &result)
{
  return reduceAdd(basis, vectors, 0.0, result);
}

} // end namespace Albany

#endif /* ALBANY_BASISOPS_HPP */
