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
GaussNewtonOperatorFactoryBase<Derived>::GaussNewtonOperatorFactoryBase(const RCP<const Epetra_MultiVector> &reducedBasis) :
  reducedBasis_(reducedBasis),
  jacobianFactory_(reducedBasis_)
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
  const int err = reduce(*this->getLeftBasis(), fullVec, result);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return result;
}

template <typename Derived>
RCP<Epetra_CrsMatrix> GaussNewtonOperatorFactoryBase<Derived>::reducedJacobianNew() {
  return jacobianFactory_.reducedMatrixNew();
}

template <typename Derived>
const Epetra_CrsMatrix &GaussNewtonOperatorFactoryBase<Derived>::reducedJacobian(Epetra_CrsMatrix &result) const {
  return jacobianFactory_.reducedMatrix(*this->getLeftBasis(), result);
}

template <typename Derived>
void GaussNewtonOperatorFactoryBase<Derived>::fullJacobianIs(const Epetra_Operator &op) {
  jacobianFactory_.fullJacobianIs(op);
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getPremultipliedReducedBasis() const {
  return jacobianFactory_.premultipliedRightProjector();
}

template <typename Derived>
RCP<const Epetra_MultiVector> GaussNewtonOperatorFactoryBase<Derived>::getLeftBasis() const {
  return static_cast<const Derived *>(this)->leftProjectorBasis();
}

GaussNewtonOperatorFactory::GaussNewtonOperatorFactory(const RCP<const Epetra_MultiVector> &reducedBasis) :
  GaussNewtonOperatorFactoryBase<GaussNewtonOperatorFactory>(reducedBasis)
{
  // Nothing to do
}

RCP<const Epetra_MultiVector> GaussNewtonOperatorFactory::leftProjectorBasis() const {
  return this->getPremultipliedReducedBasis();
}

GaussNewtonMetricOperatorFactory::GaussNewtonMetricOperatorFactory(
    const RCP<const Epetra_MultiVector> &reducedBasis,
    const Teuchos::RCP<const Epetra_Operator> &metric) :
  GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory>(reducedBasis),
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
