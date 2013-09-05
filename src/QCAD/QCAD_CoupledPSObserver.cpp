//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Epetra_MultiVector.h"

#include "Albany_Application.hpp"
#include "QCAD_CoupledPSObserver.hpp"



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OBSERVER
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

QCAD::CoupledPS_NOXObserver::CoupledPS_NOXObserver(const Teuchos::RCP<CoupledPoissonSchrodinger> &psModel) : 
  psModel_(psModel)
{
   // Nothing to do
}

void QCAD::CoupledPS_NOXObserver::observeSolution(const Epetra_Vector& solution)
{
  // Use time == 0  when none is given
  observeSolution(solution,  0.0);
}

void QCAD::CoupledPS_NOXObserver::observeSolution(
       const Epetra_Vector& solution, double time_or_param_val)
{
  Teuchos::RCP<const Epetra_Vector> soln_poisson, soln_eigenvals_dist;
  Teuchos::RCP<const Epetra_MultiVector> soln_schrodinger;

  psModel_->separateCombinedVector(Teuchos::rcp(&solution, false), 
				   soln_poisson, soln_schrodinger, soln_eigenvals_dist);

  int nEigenvals = soln_eigenvals_dist->Map().NumGlobalElements();
  Teuchos::RCP<Albany::Application> poisson_app = psModel_->getPoissonApp();      
  Teuchos::RCP<Albany::Application> schrodinger_app = psModel_->getSchrodingerApp();      

  // Evaluate state field managers
  poisson_app->evaluateStateFieldManager(time_or_param_val, NULL, *soln_poisson);
  for(int i=0; i<nEigenvals; i++)
    schrodinger_app->evaluateStateFieldManager(time_or_param_val, NULL, *((*soln_schrodinger)(i)) );

  // This must come at the end since it renames the New state 
  // as the Old state in preparation for the next step
  poisson_app->getStateMgr().updateStates();
  schrodinger_app->getStateMgr().updateStates();

  /* GAH Note:
   * If solution == "Steady" or "Continuation", we need to update the solution from the initial guess prior to
   * writing it out, or we will not get the proper state of things like "Stress" in the Exodus file.
   */ 

  Epetra_Vector *poisson_ovlp_solution = poisson_app->getAdaptSolMgr()->getOverlapSolution(*soln_poisson);
  poisson_app->getDiscretization()->writeSolution(*poisson_ovlp_solution, time_or_param_val, true); // soln is overlapped

  for(int i=0; i<nEigenvals; i++) {
    Epetra_Vector *schrodinger_ovlp_solution = schrodinger_app->getAdaptSolMgr()->getOverlapSolution(*((*soln_schrodinger)(i)));
    schrodinger_app->getDiscretization()->writeSolution(*schrodinger_ovlp_solution, time_or_param_val + i*0.1, true); // soln is overlapped
  }

  soln_eigenvals_dist->Print(std::cout << "Coupled PS Solution Eigenvalues:" << std::endl);

  //Test: use discretization built by coupled poisson-schrodinger model, which has separated solution vector specified in input file
  psModel_->getDiscretization()->writeSolution(solution, time_or_param_val, false); // soln is non-overlapped
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FACTORY
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

QCAD::CoupledPS_NOXObserverFactory::CoupledPS_NOXObserverFactory(const Teuchos::RCP<CoupledPoissonSchrodinger> &psModel) :
  psModel_(psModel)
{}

Teuchos::RCP<NOX::Epetra::Observer>
QCAD::CoupledPS_NOXObserverFactory::createInstance()
{
  Teuchos::RCP<NOX::Epetra::Observer> result(new QCAD::CoupledPS_NOXObserver(psModel_));
  //#ifdef ALBANY_MOR
  //TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "MOR observers not supported by QCAD's coupled P-S solver");
  //#endif
  return result;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<NOX::Epetra::Observer>
QCAD::CoupledPS_NOXObserverConstructor::getInstance(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  if (Teuchos::is_null(instance_)) {
    instance_ = factory_.createInstance();
  }
  return instance_;
}

