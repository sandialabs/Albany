//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_PETROVGALERKINOPERATORFACTOR_HPP
#define MOR_PETROVGALERKINOPERATORFACTOR_HPP

#include "MOR_ReducedOperatorFactory.hpp"

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_CrsMatrix;

#include "MOR_ReducedJacobianFactory.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

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

} // namespace MOR

#endif /* MOR_PETROVGALERKINOPERATORFACTOR_HPP */
