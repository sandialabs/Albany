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


#include "Albany_NOXObserver.hpp"

#include "Albany_StateManager.hpp"

Albany_NOXObserver::Albany_NOXObserver(
				       const Teuchos::RCP<Albany::Application> &app_) : 
  app(app_),
  exodusOutput(app_->getDiscretization())
{
   // Nothing to do
}

void Albany_NOXObserver::observeSolution(
					 const Epetra_Vector& solution)
{
  // Use time as time_or_param_val  when none is given
  if ( app->getParamLib()->isParameter("Time") )
    observeSolution(solution,  app->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>("Time"));
  else
    observeSolution(solution,  0.0);
}

void Albany_NOXObserver::observeSolution(
       const Epetra_Vector& solution, double time_or_param_val)
{

//#ifdef ALBANY_SEACAS
//  if(app->getSolutionMethod() != Albany::Application::Steady){
////    exodusOutput.writeSolution(time_or_param_val, solution); // soln is not overlapped
//    Epetra_Vector *ovlp_solution = app->getOverlapSolution(solution);
//    exodusOutput.writeSolution(time_or_param_val, *ovlp_solution, true); // soln is overlapped
//  }
//#endif

  // Evaluate state field manager
  app->evaluateStateFieldManager(time_or_param_val, NULL, solution);

  // This must come at the end since it renames the New state 
  // as the Old state in preparation for the next step
  app->getStateMgr().updateStates();

  /* GAH Note:
   * If solution == "Steady", we need to update the solution from the initial guess prior to
   * writing it out, or we will not get the proper state of things like "Stress" in the Exodus file.
   */ 

#ifdef ALBANY_SEACAS
//  if(app->getSolutionMethod() == Albany::Application::Steady){
//    exodusOutput.writeSolution(time_or_param_val, solution); // soln is not overlapped
    Epetra_Vector *ovlp_solution = app->getOverlapSolution(solution);
    exodusOutput.writeSolution(time_or_param_val, *ovlp_solution, true); // soln is overlapped
//  }
#endif
}
