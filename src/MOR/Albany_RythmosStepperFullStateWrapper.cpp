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

#include "Albany_RythmosStepperFullStateWrapper.hpp"

#include "Albany_ReducedSpace.hpp"
#include "Albany_RythmosUtils.hpp"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_TestForException.hpp"

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::null;
using ::Teuchos::Array;

RythmosStepperFullStateWrapper::RythmosStepperFullStateWrapper(
    const RCP<const Rythmos::StepperBase<double> > &wrappedStepper,
    const RCP<const ReducedSpace> &reducedSpace,
    const RCP<const Epetra_Map> &fullMap) :
  wrappedStepper_(wrappedStepper),
  reducedSpace_(reducedSpace),
  fullMap_(fullMap)
{
  // Nothing to do
}

void RythmosStepperFullStateWrapper::setParameterList(
    const Teuchos::RCP<Teuchos::ParameterList> &/*paramList*/) {
  this->failNonconstFunction();
}

RCP<Teuchos::ParameterList> RythmosStepperFullStateWrapper::getNonconstParameterList() {
  this->failNonconstFunction();
  return null;
}

RCP<Teuchos::ParameterList> RythmosStepperFullStateWrapper::unsetParameterList() {
  this->failNonconstFunction();
  return null;
}

RCP<const Thyra::VectorSpaceBase<double> > RythmosStepperFullStateWrapper::get_x_space() const {
  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not implemented");
  return wrappedStepper_->get_x_space(); // TODO
}

void RythmosStepperFullStateWrapper::addPoints(
    const Array<double> &/*time_vec*/,
    const Array<RCP<const Thyra::VectorBase<double> > > &/*x_vec*/,
    const Array<RCP<const Thyra::VectorBase<double> > > &/*xdot_vec*/) {
  this->failNonconstFunction();
}

void RythmosStepperFullStateWrapper::getPoints(
    const Array<double> &time_vec,
    Array<RCP<const Thyra::VectorBase<double> > > *x_vec,
    Array<RCP<const Thyra::VectorBase<double> > > *xdot_vec,
    Array<ScalarMag> *accuracy_vec) const {
  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not implemented");
  wrappedStepper_->getPoints(time_vec, x_vec, xdot_vec, accuracy_vec); // TODO
}

void RythmosStepperFullStateWrapper::getNodes(Array<double> *time_vec) const {
  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not implemented");
  wrappedStepper_->getNodes(time_vec); // TODO
}

void RythmosStepperFullStateWrapper::removeNodes(Array<double> &/*time_vec*/) {
  this->failNonconstFunction();
}

int RythmosStepperFullStateWrapper::getOrder() const {
  return wrappedStepper_->getOrder();
}

Rythmos::TimeRange<double> RythmosStepperFullStateWrapper::getTimeRange() const {
  return wrappedStepper_->getTimeRange();
}

bool RythmosStepperFullStateWrapper::supportsCloning() const {
  return true;
}

RCP<Rythmos::StepperBase<double> > RythmosStepperFullStateWrapper::cloneStepperAlgorithm() const {
  return rcp(new RythmosStepperFullStateWrapper(*this));
}

bool RythmosStepperFullStateWrapper::isImplicit() const {
  return wrappedStepper_->isImplicit();
}

bool RythmosStepperFullStateWrapper::acceptsModel() const {
  return false;
}

void RythmosStepperFullStateWrapper::setModel(const RCP<const Thyra::ModelEvaluator<double> > &/*model*/) {
  this->failNonconstFunction();
}

void RythmosStepperFullStateWrapper::setNonconstModel(const RCP<Thyra::ModelEvaluator<double> > &/*model*/) {
  this->failNonconstFunction();
}

bool RythmosStepperFullStateWrapper::modelIsConst() const {
  return true;
}

RCP<const Thyra::ModelEvaluator<double> > RythmosStepperFullStateWrapper::getModel() const {
  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not implemented");
  return wrappedStepper_->getModel(); // TODO
}

RCP<Thyra::ModelEvaluator<double> > RythmosStepperFullStateWrapper::getNonconstModel() {
  this->failNonconstFunction();
  return null;
}

void RythmosStepperFullStateWrapper::setInitialCondition(const Thyra::ModelEvaluatorBase::InArgs<double>& /*initialCondition*/) {
  this->failNonconstFunction();
}

Thyra::ModelEvaluatorBase::InArgs<double> RythmosStepperFullStateWrapper::getInitialCondition() const {
  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not implemented");
  return wrappedStepper_->getInitialCondition(); // TODO
}

double RythmosStepperFullStateWrapper::takeStep(double /*dt*/, Rythmos::StepSizeType /*stepType*/) {
  this->failNonconstFunction();
  return 0.0;
}

const Rythmos::StepStatus<double> RythmosStepperFullStateWrapper::getStepStatus() const {
  Rythmos::StepStatus<double> result = wrappedStepper_->getStepStatus();

  const RCP<const Thyra::VectorBase<double> > reducedSolution = getRythmosState(result.solution);
  const RCP<const Thyra::VectorBase<double> > reducedSolutionDot = getRythmosState(result.solutionDot);

  const Epetra_Map &reducedMap = reducedSpace_->componentMap();
  const RCP<const Epetra_Vector> epetraReducedSolution = Thyra::get_Epetra_Vector(reducedMap, reducedSolution);
  const RCP<const Epetra_Vector> epetraReducedSolutionDot = Thyra::get_Epetra_Vector(reducedMap, reducedSolutionDot);

  const RCP<const Epetra_Vector> epetraFullSolution = reducedSpace_->expansion(*epetraReducedSolution);
  const RCP<const Epetra_Vector> epetraFullSolutionDot = reducedSpace_->expansion(*epetraReducedSolutionDot);

  const RCP<const Thyra::VectorSpaceBase<double> > fullVectorSpace = Thyra::create_VectorSpace(fullMap_);
  const RCP<const Thyra::VectorBase<double> > fullSolution = Thyra::create_Vector(epetraFullSolution, fullVectorSpace);
  const RCP<const Thyra::VectorBase<double> > fullSolutionDot = Thyra::create_Vector(epetraFullSolutionDot, fullVectorSpace);

  result.solution = Thyra::create_Vector(epetraFullSolution, fullVectorSpace);
  result.solutionDot = Thyra::create_Vector(epetraFullSolutionDot, fullVectorSpace);
  result.residual = null; // Residual expansion not implemented
  return result;
}

void RythmosStepperFullStateWrapper::setStepControlData(const Rythmos::StepperBase<double> &/*stepper*/) {
  this->failNonconstFunction();
}

void RythmosStepperFullStateWrapper::failNonconstFunction() {
  TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Wrapped object is const");
}

} // namespace Albany
