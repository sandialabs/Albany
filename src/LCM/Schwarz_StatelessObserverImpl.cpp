//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_StatelessObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"

#include <string>

//#define OUTPUT_TO_SCREEN 

namespace LCM {

StatelessObserverImpl::
StatelessObserverImpl(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> &apps)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  apps_ = apps;
  n_models_ = apps.size();
  sol_out_time_ = Teuchos::TimeMonitor::getNewTimer("Albany: Output to File");
}

RealType StatelessObserverImpl::
getTimeParamValueOrDefault(RealType default_value) const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  // FIXME, IKT, : We may want to change the logic here at some point.
  // I am assuming all the models have the same parameters,
  // so we only pull the time-label from the 0th model.
  const std::string label("Time");
  return
      (apps_[0]->getParamLib()->isParameter(label)) ?
          apps_[0]->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(
              label) :
          default_value;
}

void StatelessObserverImpl::observeSolutionT(
    double stamp,
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector>> non_overlapped_solutionT,
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector>> non_overlapped_solution_dotT)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  Teuchos::TimeMonitor timer(*sol_out_time_);
  for (int m = 0; m < n_models_; m++) {
    const Teuchos::RCP<const Tpetra_Vector> overlapped_solutionT =
        apps_[m]->getOverlapSolutionT(*non_overlapped_solutionT[m]);
    apps_[m]->getDiscretization()->writeSolutionT(
        *overlapped_solutionT, stamp, /*overlapped =*/true);
  }
}

} // namespace LCM
