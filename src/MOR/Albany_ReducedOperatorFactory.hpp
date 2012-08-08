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

#ifndef ALBANY_REDUCEDOPERATORFACTOR_HPP
#define ALBANY_REDUCEDOPERATORFACTOR_HPP

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_CrsMatrix;

#include "Teuchos_RCP.hpp"

namespace Albany {

class ReducedOperatorFactory {
public:
  ReducedOperatorFactory();

  virtual bool fullJacobianRequired(bool residualRequested, bool jacobianRequested) const = 0;

  virtual const Epetra_MultiVector &leftProjection(const Epetra_MultiVector &fullVector,
                                                   Epetra_MultiVector &result) const = 0;

  virtual Teuchos::RCP<Epetra_CrsMatrix> reducedJacobianNew() = 0;
  virtual const Epetra_CrsMatrix &reducedJacobian(Epetra_CrsMatrix &result) const = 0;

  virtual void fullJacobianIs(const Epetra_Operator &op) = 0;

  virtual ~ReducedOperatorFactory();

private:
  // Disallow copy and assignment
  ReducedOperatorFactory(const ReducedOperatorFactory &);
  ReducedOperatorFactory &operator=(const ReducedOperatorFactory &);
};

inline
ReducedOperatorFactory::ReducedOperatorFactory() {
  // Nothing to do
}

inline
ReducedOperatorFactory::~ReducedOperatorFactory() {
  // Nothing to do
}

} // namespace Albany

#endif /* ALBANY_REDUCEDOPERATORFACTOR_HPP */
