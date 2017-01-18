//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_GaussNewtonOperatorFactory.hpp"

#include "MOR_BasisOps.hpp"

#include "Epetra_Operator.h"
#include "Epetra_Map.h"

namespace MOR {

using ::Teuchos::RCP;

template <typename Derived>
GaussNewtonOperatorFactoryBase<Derived>::GaussNewtonOperatorFactoryBase(const RCP<const Epetra_MultiVector> &reducedBasis, int numDBCModes) :
  reducedBasis_(reducedBasis),
  jacobianFactory_(reducedBasis_)
  ,num_dbc_modes_(numDBCModes)
{
  // Nothing to do
}

template <typename Derived>
bool GaussNewtonOperatorFactoryBase<Derived>::fullJacobianRequired(bool residualRequested, bool jacobianRequested) const {
  return residualRequested || jacobianRequested;
}

template <typename Derived>
const Epetra_MultiVector &GaussNewtonOperatorFactoryBase<Derived>::leftProjection(
    const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const {
  int err = 0;
  if (num_dbc_modes_ == 0)
    err = reduce(*this->getLeftBasis(), fullVec, result);
  else
    err = reduce(*this->getLeftBasisCopy(), fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

template <typename Derived>
RCP<Epetra_CrsMatrix> GaussNewtonOperatorFactoryBase<Derived>::reducedJacobianNew() {
  return jacobianFactory_.reducedMatrixNew();
}

template <typename Derived>
const Epetra_CrsMatrix &GaussNewtonOperatorFactoryBase<Derived>::reducedJacobian(Epetra_CrsMatrix &result) const {
  if (num_dbc_modes_ == 0)
    return jacobianFactory_.reducedMatrix(*this->getLeftBasis(), result);
  else
    return jacobianFactory_.reducedMatrix(*this->getLeftBasisCopy(), result);
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::fullJacobianIs(const Epetra_Operator &op) {
  jacobianFactory_.fullJacobianIs(op);

  leftbasis_ = Teuchos::rcp(new Epetra_MultiVector(*jacobianFactory_.premultipliedRightProjector()));

  //printf("using %d DBC modes\n",num_dbc_modes_);
  if (num_dbc_modes_ > 0)
  {
    Epetra_MultiVector* psi_dbc = new Epetra_MultiVector(View,*leftbasis_,0,num_dbc_modes_);
  //psi_dbc->Print(std::cout);
    Epetra_MultiVector* phi_dbc = new Epetra_MultiVector(View,*jacobianFactory_.rightProjector(),0,num_dbc_modes_);
    psi_dbc->Scale(1.0, *phi_dbc);
  //psi_dbc->Print(std::cout);
    delete psi_dbc;
    delete phi_dbc;
  }
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getPremultipliedReducedBasis() const {
  return jacobianFactory_.premultipliedRightProjector();
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getLeftBasis() const {
  return static_cast<const Derived *>(this)->leftProjectorBasis();
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getLeftBasisCopy() const {
  return leftbasis_;
}

GaussNewtonOperatorFactory::GaussNewtonOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis, int numDBCModes) :
  GaussNewtonOperatorFactoryBase<GaussNewtonOperatorFactory>(reducedBasis, numDBCModes)
{
  // Nothing to do
}

RCP<const Epetra_MultiVector> GaussNewtonOperatorFactory::leftProjectorBasis() const {
  return this->getPremultipliedReducedBasis();
}

GaussNewtonMetricOperatorFactory::GaussNewtonMetricOperatorFactory(
    const RCP<const Epetra_MultiVector> &reducedBasis,
    const Teuchos::RCP<const Epetra_Operator> &metric) :
  GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory>(reducedBasis, num_dbc_modes_),
  metric_(metric),
  premultipliedLeftProjector_(new Epetra_MultiVector(metric->OperatorDomainMap(), reducedBasis->NumVectors(), false))
{
  this->updatePremultipliedLeftProjector();
}

RCP<const Epetra_MultiVector> GaussNewtonMetricOperatorFactory::leftProjectorBasis() const {
  return this->premultipliedLeftProjector_;
}

void GaussNewtonMetricOperatorFactory::fullJacobianIs(const Epetra_Operator &op) {
  this->GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory>::fullJacobianIs(op);
  this->updatePremultipliedLeftProjector();
}

void GaussNewtonMetricOperatorFactory::updatePremultipliedLeftProjector() {
  const int err = metric_->Apply(*this->getPremultipliedReducedBasis(), *premultipliedLeftProjector_);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
}

} // namespace MOR
