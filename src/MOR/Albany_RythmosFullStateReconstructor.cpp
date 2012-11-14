//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_RythmosFullStateReconstructor.hpp"

#include "Albany_ReducedSpace.hpp"
#include "Albany_BasisInputFile.hpp"

#include "Albany_RythmosStepperFullStateWrapper.hpp"

#include "Epetra_Vector.h"
#include "Epetra_Map.h"

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::rcpFromRef;
using ::Teuchos::ParameterList;

RythmosFullStateReconstructor::RythmosFullStateReconstructor(
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &decoratedObserver,
    const Teuchos::RCP<const Epetra_Map> &decoratedMap) :
  decoratedObserver_(decoratedObserver),
  decoratedMap_(decoratedMap),
  reducedSpace_()
{
  fillDefaultBasisInputParams(params);
  const RCP<const Epetra_MultiVector> orthogonalBasis = readOrthonormalBasis(*decoratedMap, params);
  reducedSpace_ = rcp(new LinearReducedSpace(*orthogonalBasis));
}

RCP<Rythmos::IntegrationObserverBase<double> > RythmosFullStateReconstructor::cloneIntegrationObserver() const {
  return Teuchos::null; // TODO
}

void RythmosFullStateReconstructor::resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain) {
  decoratedObserver_->resetIntegrationObserver(integrationTimeDomain);
}

void RythmosFullStateReconstructor::observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  const RythmosStepperFullStateWrapper fullStepper(rcpFromRef(stepper), reducedSpace_, decoratedMap_);
  decoratedObserver_->observeStartTimeStep(fullStepper, stepCtrlInfo, timeStepIter);
}

void RythmosFullStateReconstructor::observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter) {
  const RythmosStepperFullStateWrapper fullStepper(rcpFromRef(stepper), reducedSpace_, decoratedMap_);
  decoratedObserver_->observeCompletedTimeStep(fullStepper, stepCtrlInfo, timeStepIter);
}

} // namespace Albany
