//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_RYTHMOSFULLSTATERECONSTRUCTOR_HPP
#define MOR_RYTHMOSFULLSTATERECONSTRUCTOR_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

namespace MOR {

class ReducedSpace;

class RythmosFullStateReconstructor : public Rythmos::IntegrationObserverBase<double> {
public:
  RythmosFullStateReconstructor(
      const Teuchos::RCP<const ReducedSpace> &reducedSpace,
      const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &decoratedObserver);

  // Overridden
  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > cloneIntegrationObserver() const;

  virtual void resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain);

  virtual void observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

  virtual void observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

private:
  Teuchos::RCP<const ReducedSpace> reducedSpace_;
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > decoratedObserver_;
};

} // namespace MOR

#endif /*MOR_RYTHMOSFULLSTATERECONSTRUCTOR_HPP*/
