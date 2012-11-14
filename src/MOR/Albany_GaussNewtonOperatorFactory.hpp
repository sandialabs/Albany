//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_GAUSSNEWTONOPERATORFACTOR_HPP
#define ALBANY_GAUSSNEWTONOPERATORFACTOR_HPP

#include "Albany_ReducedOperatorFactory.hpp"

class Epetra_MultiVector;
class Epetra_CrsMatrix;
class Epetra_Operator;

#include "Albany_ReducedJacobianFactory.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

template <typename Derived>
class GaussNewtonOperatorFactoryBase : public ReducedOperatorFactory {
public:
  explicit GaussNewtonOperatorFactoryBase(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis);

  virtual bool fullJacobianRequired(bool residualRequested, bool jacobianRequested) const;

  virtual const Epetra_MultiVector &leftProjection(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const;

  virtual Teuchos::RCP<Epetra_CrsMatrix> reducedJacobianNew();
  virtual const Epetra_CrsMatrix &reducedJacobian(Epetra_CrsMatrix &result) const;

  virtual void fullJacobianIs(const Epetra_Operator &op);

protected:
  Teuchos::RCP<const Epetra_MultiVector> getPremultipliedReducedBasis() const;

private:
  Teuchos::RCP<const Epetra_MultiVector> reducedBasis_;

  ReducedJacobianFactory jacobianFactory_;

  Teuchos::RCP<const Epetra_MultiVector> getLeftBasis() const;
};

class GaussNewtonOperatorFactory : public GaussNewtonOperatorFactoryBase<GaussNewtonOperatorFactory> {
public:
  explicit GaussNewtonOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis);

  Teuchos::RCP<const Epetra_MultiVector> leftProjectorBasis() const;
};

class GaussNewtonMetricOperatorFactory : public GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory> {
public:
  GaussNewtonMetricOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis,
                                   const Teuchos::RCP<const Epetra_Operator> &metric);

  // Overridden
  virtual void fullJacobianIs(const Epetra_Operator &op);

  Teuchos::RCP<const Epetra_MultiVector> leftProjectorBasis() const;

private:
  Teuchos::RCP<const Epetra_Operator> metric_;

  Teuchos::RCP<Epetra_MultiVector> premultipliedLeftProjector_;

  void updatePremultipliedLeftProjector();
};

} // namespace Albany

#endif /* ALBANY_GAUSSNEWTONOPERATORFACTOR_HPP */
