//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: right now this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#ifndef ALBANY_RYTHMOSOBSERVER
#define ALBANY_RYTHMOSOBSERVER

#include "Rythmos_IntegrationObserverBase.hpp"

#include "Albany_Application.hpp"
#include "Albany_ObserverImpl.hpp"

class Albany_RythmosObserver : public Rythmos::IntegrationObserverBase<RealType>
{
public:
   Albany_RythmosObserver (
         const Teuchos::RCP<Albany::Application> &app_);

   ~Albany_RythmosObserver ()
   { };

  typedef RealType ScalarType;

  Teuchos::RCP<Rythmos::IntegrationObserverBase<ScalarType> >
    cloneIntegrationObserver() const
  {  TEUCHOS_TEST_FOR_EXCEPT(true);};

  void resetIntegrationObserver(
    const Rythmos::TimeRange<ScalarType> &integrationTimeDomain
    )
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
   Albany::ObserverImpl impl;

   bool initial_step;

};

#endif //ALBANY_RYTHMOSOBSERVER
