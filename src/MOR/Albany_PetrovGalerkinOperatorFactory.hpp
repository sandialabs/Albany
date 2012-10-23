//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_PETROVGALERKINOPERATORFACTOR_HPP
#define ALBANY_PETROVGALERKINOPERATORFACTOR_HPP

#include "Albany_ReducedOperatorFactory.hpp"

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_CrsMatrix;

#include "Albany_ReducedJacobianFactory.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

class PetrovGalerkinOperatorFactory : public ReducedOperatorFactory {
public:
  explicit PetrovGalerkinOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis);
  PetrovGalerkinOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis,
                                const Teuchos::RCP<const Epetra_MultiVector> &projectionBasis);

  virtual bool fullJacobianRequired(bool residualRequested, bool jacobianRequested) const;

  virtual const Epetra_MultiVector &leftProjection(const Epetra_MultiVector &fullVector,
                                                   Epetra_MultiVector &result) const;

  virtual Teuchos::RCP<Epetra_CrsMatrix> reducedJacobianNew();
  virtual const Epetra_CrsMatrix &reducedJacobian(Epetra_CrsMatrix &result) const;

  virtual void fullJacobianIs(const Epetra_Operator &op);

private:
  Teuchos::RCP<const Epetra_MultiVector> reducedBasis_, projectionBasis_;

  ReducedJacobianFactory jacobianFactory_;
};

} // namespace Albany

#endif /* ALBANY_PETROVGALERKINOPERATORFACTOR_HPP */
