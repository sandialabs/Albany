//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDOPERATORFACTOR_HPP
#define MOR_REDUCEDOPERATORFACTOR_HPP

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_CrsMatrix;

#include "Teuchos_RCP.hpp"

namespace MOR {

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

} // namespace MOR

#endif /* MOR_REDUCEDOPERATORFACTOR_HPP */
