//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_RythmosProjectionErrorObserver.hpp"

#include "MOR_RythmosUtils.hpp"

#include "Epetra_Vector.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

RythmosProjectionErrorObserver::RythmosProjectionErrorObserver(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile) :
  projectionError_(projectionSpace, errorFile)
{
  // Nothing to do
}

Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > RythmosProjectionErrorObserver::cloneIntegrationObserver() const {
  return Teuchos::null; // TODO Enable cloning
}

void RythmosProjectionErrorObserver::resetIntegrationObserver(const Rythmos::TimeRange<double> &/*integrationTimeDomain*/) {
  // Not implemented
}

void RythmosProjectionErrorObserver::observeStartTimeIntegration(
    const Rythmos::StepperBase<double> &stepper) {
  this->observeTimeStep(stepper);
}

void RythmosProjectionErrorObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &/*stepCtrlInfo*/,
    const int /*timeStepIter*/) {
  this->observeTimeStep(stepper);
}

void RythmosProjectionErrorObserver::observeTimeStep(
    const Rythmos::StepperBase<double> &stepper) {
  const Rythmos::StepStatus<double> stepStatus = stepper.getStepStatus();

  const Teuchos::RCP<const Thyra::VectorBase<double> > stepperSolution = stepStatus.solution;
  const Teuchos::RCP<const Thyra::VectorBase<double> > stepperState = getRythmosState(stepperSolution);
  const Teuchos::RCP<const Thyra::VectorSpaceBase<double> > stateSpace = stepperState->space();

  const Teuchos::RCP<const Epetra_Comm> stateComm = Teuchos::rcpFromRef(projectionError_.projectionBasisComm());
  const Teuchos::RCP<const Epetra_Map> stateMap = Thyra::get_Epetra_Map(*stateSpace, stateComm);
  const Teuchos::RCP<const Epetra_Vector> state = Thyra::get_Epetra_Vector(*stateMap, stepperState);

  projectionError_.process(*state);
}

} // namespace MOR
