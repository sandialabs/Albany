//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_RYTHMOSFULLSTATERECONSTRUCTOR_HPP
#define ALBANY_RYTHMOSFULLSTATERECONSTRUCTOR_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "Teuchos_ParameterList.hpp"

class Epetra_Map;

namespace Albany {

class ReducedSpace;

class RythmosFullStateReconstructor : public Rythmos::IntegrationObserverBase<double> {
public:
  explicit RythmosFullStateReconstructor(const Teuchos::RCP<Teuchos::ParameterList> &params,
                                         const Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > &decoratedObserver,
                                         const Teuchos::RCP<const Epetra_Map> &decoratedMap);

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
  Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > decoratedObserver_;
  Teuchos::RCP<const Epetra_Map> decoratedMap_;
  Teuchos::RCP<const ReducedSpace> reducedSpace_;
};

} // namespace Albany

#endif /*ALBANY_RYTHMOSFULLSTATERECONSTRUCTOR_HPP*/
