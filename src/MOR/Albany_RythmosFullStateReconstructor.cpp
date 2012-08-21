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
