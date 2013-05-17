//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_RYTHMOSOBSERVER
#define ALBANY_RYTHMOSOBSERVER

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Rythmos_StepperBase.hpp"
#include "Rythmos_IntegrationObserverBase.hpp"
#include "Rythmos_TimeRange.hpp"
#include "Albany_Application.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Teuchos_TimeMonitor.hpp"

typedef double Scalar;

class Albany_RythmosObserver : public Rythmos::IntegrationObserverBase<Scalar>
{
public:
   Albany_RythmosObserver (
         const Teuchos::RCP<Albany::Application> &app_);

   ~Albany_RythmosObserver ()
   { };


  Teuchos::RCP<Rythmos::IntegrationObserverBase<Scalar> >
    cloneIntegrationObserver() const
  {  TEUCHOS_TEST_FOR_EXCEPT(true);};

  void resetIntegrationObserver(
    const Rythmos::TimeRange<Scalar> &integrationTimeDomain
    )
  { };

  // Print initial condition
  void observeStartTimeStep(
    const Rythmos::StepperBase<Scalar> &stepper,
    const Rythmos::StepControlInfo<Scalar> &stepCtrlInfo,
    const int timeStepIter
    );

  void observeCompletedTimeStep(
    const Rythmos::StepperBase<Scalar> &stepper,
    const Rythmos::StepControlInfo<Scalar> &stepCtrlInfo,
    const int timeStepIter
    );

private:
   Teuchos::RCP<Albany::AbstractDiscretization> disc;
   Teuchos::RCP<Albany::Application> app;

   bool initial_step;

};

#endif //ALBANY_RYTHMOSOBSERVER
