//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_RythmosSnapshotCollectionObserver.hpp"

#include "MOR_RythmosUtils.hpp"

#include "Epetra_Vector.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

using ::Teuchos::RCP;
using ::Teuchos::ParameterList;

RythmosSnapshotCollectionObserver::RythmosSnapshotCollectionObserver(
    int period,
    Teuchos::RCP<MultiVectorOutputFile> snapshotFile) :
  snapshotCollector_(period, snapshotFile)
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
    const Rythmos::StepperBase<double> &stepper) {
  const Rythmos::StepStatus<double> stepStatus = stepper.getStepStatus();

  const Teuchos::RCP<const Thyra::VectorBase<double> > stepperSolution = stepStatus.solution;
  const Teuchos::RCP<const Thyra::VectorBase<double> > stepperState = getRythmosState(stepperSolution);
  const Teuchos::RCP<const Thyra::VectorSpaceBase<double> > stateSpace = stepperState->space();
  const Teuchos::RCP<const Teuchos::Comm<Thyra::Ordinal> > stateSpaceComm = getComm(*stateSpace);

  const Teuchos::RCP<const Epetra_Comm> stateComm = Thyra::get_Epetra_Comm(*stateSpaceComm);
  const Teuchos::RCP<const Epetra_Map> stateMap = Thyra::get_Epetra_Map(*stateSpace, stateComm);
  const Teuchos::RCP<const Epetra_Vector> state = Thyra::get_Epetra_Vector(*stateMap, stepperState);

  const double stamp = stepStatus.time;

  snapshotCollector_.addVector(stamp, *state);
}

void RythmosSnapshotCollectionObserver::observeStartTimeIntegration(
    const Rythmos::StepperBase<double> &stepper) {
  this->observeTimeStep(stepper);
}

void RythmosSnapshotCollectionObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &/*stepCtrlInfo*/,
    const int /*timeStepIter*/) {
  this->observeTimeStep(stepper);
}

} // namespace MOR
