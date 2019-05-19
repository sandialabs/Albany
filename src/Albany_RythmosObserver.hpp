//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_RYTHMOS_OBSERVER_HPP
#define ALBANY_RYTHMOS_OBSERVER_HPP

#include "Rythmos_IntegrationObserverBase.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace Albany
{
class Application;
class ObserverImpl;

class RythmosObserver : public Rythmos::IntegrationObserverBase<RealType>
{
public:
   RythmosObserver (const Teuchos::RCP<Application> &app_);

   ~RythmosObserver () = default;

  typedef RealType ScalarType;

  Teuchos::RCP<Rythmos::IntegrationObserverBase<ScalarType> >
  cloneIntegrationObserver() const {  TEUCHOS_TEST_FOR_EXCEPT(true);}

  void resetIntegrationObserver(const Rythmos::TimeRange<ScalarType>& /* integrationTimeDomain */)
  { };

  // Print initial condition
  void observeStartTimeStep(
    const Rythmos::StepperBase<ScalarType> &stepper,
    const Rythmos::StepControlInfo<ScalarType> &stepCtrlInfo,
    const int timeStepIter
    );

  void observeCompletedTimeStep(
    const Rythmos::StepperBase<ScalarType> &stepper,
    const Rythmos::StepControlInfo<ScalarType> &stepCtrlInfo,
    const int timeStepIter
    );

private:
   Teuchos::RCP<ObserverImpl> impl;

   bool initial_step;
};

} // namespace Albany

#endif // ALBANY_RYTHMOS_OBSERVER_HPP
