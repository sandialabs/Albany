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


#ifndef ALBANY_RYTHMOSOBSERVER
#define ALBANY_RYTHMOSOBSERVER

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Rythmos_StepperBase.hpp"
#include "Rythmos_IntegrationObserverBase.hpp"
#include "Rythmos_TimeRange.hpp"
#include "Albany_Application.hpp"
#include "Albany_ExodusOutput.hpp"
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

   Albany::ExodusOutput exodusOutput;
};

#endif //ALBANY_RYTHMOSOBSERVER
