//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_RythmosFullStateReconstructor.hpp"

#include "MOR_ReducedSpace.hpp"

#include "MOR_RythmosStepperFullStateWrapper.hpp"

namespace MOR {

RythmosFullStateReconstructor::RythmosFullStateReconstructor(
    const Teuchos::RCP<const ReducedSpace> &reducedSpace,
    const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &decoratedObserver) :
  reducedSpace_(reducedSpace),
  decoratedObserver_(decoratedObserver)
{
  // Nothing to do
}

Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > RythmosFullStateReconstructor::cloneIntegrationObserver() const {
  return Teuchos::null; // TODO
}

void RythmosFullStateReconstructor::resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain) {
  decoratedObserver_->resetIntegrationObserver(integrationTimeDomain);
}

void RythmosFullStateReconstructor::observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  const RythmosStepperFullStateWrapper fullStepper(Teuchos::rcpFromRef(stepper), reducedSpace_);
  decoratedObserver_->observeStartTimeStep(fullStepper, stepCtrlInfo, timeStepIter);
}

void RythmosFullStateReconstructor::observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  const RythmosStepperFullStateWrapper fullStepper(Teuchos::rcpFromRef(stepper), reducedSpace_);
  decoratedObserver_->observeCompletedTimeStep(fullStepper, stepCtrlInfo, timeStepIter);
}

} // namespace MOR
