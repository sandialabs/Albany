//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_RythmosSnapshotCollectionObserver.hpp"

#include "Albany_RythmosUtils.hpp"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::ParameterList;

RythmosSnapshotCollectionObserver::RythmosSnapshotCollectionObserver(
    const RCP<ParameterList> &params,
    const RCP<const Epetra_Map> &stateMap) :
  snapshotCollector_(params),
  stateMap_(stateMap)
{
  // Nothing to do
}

RCP<Rythmos::IntegrationObserverBase<double> > RythmosSnapshotCollectionObserver::cloneIntegrationObserver() const {
  return Teuchos::null; // TODO
}

void RythmosSnapshotCollectionObserver::resetIntegrationObserver(const Rythmos::TimeRange<double> &/*integrationTimeDomain*/) {
  // Not implemented
}

void RythmosSnapshotCollectionObserver::observeTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &/*stepCtrlInfo*/,
    const int /*timeStepIter*/) {
  const Rythmos::StepStatus<double> stepStatus = stepper.getStepStatus();

  const RCP<const Thyra::VectorBase<double> > stepperSolution = stepStatus.solution;
  const RCP<const Thyra::VectorBase<double> > stepperState = getRythmosState(stepperSolution);
  const RCP<const Epetra_Vector> state = Thyra::get_Epetra_Vector(*stateMap_, stepperState);

  const double stamp = stepStatus.time;

  snapshotCollector_.addVector(stamp, *state);
}

void RythmosSnapshotCollectionObserver::observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  this->observeTimeStep(stepper, stepCtrlInfo, timeStepIter);
}

void RythmosSnapshotCollectionObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  this->observeTimeStep(stepper, stepCtrlInfo, timeStepIter);
}

} // namespace Albany
