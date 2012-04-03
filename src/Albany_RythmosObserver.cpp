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


#include "Albany_RythmosObserver.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_VectorBase.hpp"
#ifdef ALBANY_SEACAS
  #include "Albany_STKDiscretization.hpp"
#endif

Albany_RythmosObserver::Albany_RythmosObserver(
     const Teuchos::RCP<Albany::Application> &app_) : 
  disc(app_->getDiscretization()),
  app(app_),
  exodusOutput(app->getDiscretization())
{
  // Nothing to do
}

void Albany_RythmosObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<Scalar> &stepper,
    const Rythmos::StepControlInfo<Scalar> &stepCtrlInfo,
    const int timeStepIter
    )
{
  //cout << "ALBANY OBSERVER CALLED step=" <<  timeStepIter 
  //     << ",  time=" << stepper.getStepStatus().time << endl;

  Teuchos::RCP<const Thyra::DefaultProductVector<double> > solnandsens = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<double> >
      (stepper.getStepStatus().solution);
  Teuchos::RCP<const Thyra::DefaultProductVector<double> > solnandsens_dot = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<double> >
      (stepper.getStepStatus().solutionDot);

  Teuchos::RCP<const Thyra::VectorBase<double> > solution;
  Teuchos::RCP<const Thyra::VectorBase<double> > solution_dot;
  if (solnandsens != Teuchos::null) {
    solution = solnandsens->getVectorBlock(0);
    solution_dot = solnandsens_dot->getVectorBlock(0);
  }
  else {
    solution = stepper.getStepStatus().solution;
    solution_dot = stepper.getStepStatus().solutionDot;
  }

  const Epetra_Vector soln= *(Thyra::get_Epetra_Vector(*disc->getMap(), solution));
  const Epetra_Vector soln_dot= *(Thyra::get_Epetra_Vector(*disc->getMap(), solution_dot));

  double t = stepper.getStepStatus().time;
  if ( app->getParamLib()->isParameter("Time") )
    t = app->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

#ifdef ALBANY_SEACAS
  exodusOutput.writeSolution(t, soln);
#endif

  // Evaluate state field manager
  app->evaluateStateFieldManager(t, &soln_dot, soln);

  app->getStateMgr().updateStates();;
}
