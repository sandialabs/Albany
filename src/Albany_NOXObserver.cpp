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

#ifdef ALBANY_SEACAS
#include "Albany_STKDiscretization.hpp"
#endif

Albany_NOXObserver::Albany_NOXObserver(
				       const Teuchos::RCP<Albany::Application> &app_) : 
  app(app_),
  disc(app_->getDiscretization())
{
  exooutTime = Teuchos::TimeMonitor::getNewTimer("Albany: Output to Exodus");
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
       const Epetra_Vector& solution, double time_or_param_val
                                         )
{
#ifdef ALBANY_SEACAS
  // if (solution.Map().Comm().MyPID()==0)
  //   cout << "Albany::NOXObserver calling exodus output " << endl;

  Albany::STKDiscretization* stkDisc =
    dynamic_cast<Albany::STKDiscretization*>(disc.get());

  {
    Teuchos::TimeMonitor exooutTimer(*exooutTime); //start timer

    stkDisc->outputToExodus(solution, time_or_param_val);
  }
#endif

  // Special output for loca runs of HTE problem
  //double mx;
  //solution.MaxValue(&mx);
  //cout << setprecision(9) << "MaxValue " << mx << endl;

  // Evaluate state field manager
  app->evaluateStateFieldManager(time_or_param_val, NULL, solution);

  // This must come at the end since it renames the New state 
  // as the Old state in preparation for the next step
  app->getStateMgr().updateStates();

}
