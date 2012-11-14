//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_RythmosProjectionErrorObserver.hpp"

#include "Albany_RythmosUtils.hpp"

#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::ParameterList;

RythmosProjectionErrorObserver::RythmosProjectionErrorObserver(
    const RCP<ParameterList> &params,
    const RCP<const Epetra_Map> &stateMap) :
  projectionError_(params, stateMap),
  stateMap_(stateMap)
{
  // Nothing to do
}

RCP<Rythmos::IntegrationObserverBase<double> > RythmosProjectionErrorObserver::cloneIntegrationObserver() const {
  return Teuchos::null; // TODO
}

void RythmosProjectionErrorObserver::resetIntegrationObserver(const Rythmos::TimeRange<double> &/*integrationTimeDomain*/) {
  // Not implemented
}

void RythmosProjectionErrorObserver::observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  this->observeTimeStep(stepper, stepCtrlInfo, timeStepIter);
}

void RythmosProjectionErrorObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  this->observeTimeStep(stepper, stepCtrlInfo, timeStepIter);
}

void RythmosProjectionErrorObserver::observeTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &/*stepCtrlInfo*/,
    const int /*timeStepIter*/) {
  const Rythmos::StepStatus<double> stepStatus = stepper.getStepStatus();

  const RCP<const Thyra::VectorBase<double> > stepperSolution = stepStatus.solution;
  const RCP<const Thyra::VectorBase<double> > stepperState = getRythmosState(stepperSolution);
  const RCP<const Epetra_Vector> state = Thyra::get_Epetra_Vector(*stateMap_, stepperState);

  projectionError_.process(*state);
}

} // namespace Albany
