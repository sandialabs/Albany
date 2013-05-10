//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_RythmosObserver.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_VectorBase.hpp"

Albany_RythmosObserver::Albany_RythmosObserver(
     const Teuchos::RCP<Albany::Application> &app_) : 
  disc(app_->getDiscretization()),
  app(app_),
  initial_step(true)
{
  // Nothing to do
}

void Albany_RythmosObserver::observeStartTimeStep(
    const Rythmos::StepperBase<Scalar> &stepper,
    const Rythmos::StepControlInfo<Scalar> &stepCtrlInfo,
    const int timeStepIter
    )
{

  if(initial_step)

    initial_step = false;

  else

    return;

  // Print the initial condition

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

  // Should be zero unless we are restarting
  double t = stepper.getStepStatus().time;
  if ( app->getParamLib()->isParameter("Time") )
    t = app->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>("Time");

  Epetra_Vector *ovlp_solution = app->getAdaptSolMgr()->getOverlapSolution(soln);
  disc->writeSolution(*ovlp_solution, t, true); // soln is overlapped

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

  Epetra_Vector *ovlp_solution = app->getAdaptSolMgr()->getOverlapSolution(soln);
  disc->writeSolution(*ovlp_solution, t, true); // soln is overlapped

  // Evaluate state field manager
  app->evaluateStateFieldManager(t, &soln_dot, soln);

  app->getStateMgr().updateStates();
}
