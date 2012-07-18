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

#include "Albany_RythmosSnapshotCollectionObserver.hpp"

#include "Albany_RythmosUtils.hpp"

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
