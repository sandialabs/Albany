//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_StatelessObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"

#include <string>

namespace LCM {

StatelessObserverImpl::
StatelessObserverImpl (Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > &apps)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  apps_ = apps;
  n_models_ = apps.size();  
  sol_out_time_ = Teuchos::TimeMonitor::getNewTimer("Albany: Output to File"); 
}

RealType StatelessObserverImpl::
getTimeParamValueOrDefault (RealType defaultValue) const {
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  //FIXME, IKT, : We may want to change the logic here at some point.  I am assuming all the 
  //models have the same parameters, so we only pull the time-label from the 0th model.
  const std::string label("Time");
  return (apps_[0]->getParamLib()->isParameter(label)) ?
    apps_[0]->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(label) :
    defaultValue;
}

void StatelessObserverImpl::observeSolutionT (
  double stamp, Teuchos::Array<Teuchos::RCP<const Tpetra_Vector >> nonOverlappedSolutionT,
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > nonOverlappedSolutionDotT)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  Teuchos::TimeMonitor timer(*sol_out_time_);
  for (int m=0; m<n_models_; m++) {
    const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionT =
      apps_[m]->getOverlapSolutionT(*nonOverlappedSolutionT[m]);
    apps_[0]->getDiscretization()->writeSolutionT(
    *overlappedSolutionT, stamp, /*overlapped =*/ true);
  }
}

} // namespace LCM
