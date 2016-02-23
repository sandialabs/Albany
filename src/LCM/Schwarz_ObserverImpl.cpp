//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_ObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Ptr.hpp"

#include <string>

//#define OUTPUT_TO_SCREEN 

namespace LCM {

ObserverImpl::
ObserverImpl(Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> &apps) :
    StatelessObserverImpl(apps)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  n_models_ = apps.size();
  apps_ = apps;
}

void ObserverImpl::observeSolutionT(
    double stamp,
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> non_overlapped_solutionT,
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> non_overlapped_solution_dotT)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  for (int m = 0; m < n_models_; m++) {
    apps_[m]->evaluateStateFieldManagerT(
        stamp,
        non_overlapped_solution_dotT[m].ptr(),
        Teuchos::null,
        *non_overlapped_solutionT[m]);
    apps_[m]->getStateMgr().updateStates();
  }
  StatelessObserverImpl::observeSolutionT(stamp, non_overlapped_solutionT,
      non_overlapped_solution_dotT);

}

} // namespace LCM

