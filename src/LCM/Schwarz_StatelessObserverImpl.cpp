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
  solOutTime_ = Teuchos::TimeMonitor::getNewTimer("Albany: Output to File"); 
}

RealType StatelessObserverImpl::
getTimeParamValueOrDefault (RealType defaultValue) const {
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  /*const std::string label("Time");
  return (app_->getParamLib()->isParameter(label)) ?
    app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(label) :
    defaultValue;
  */
}

Teuchos::RCP<const Tpetra_Map>
StatelessObserverImpl::getNonOverlappedMapT () const {
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  //return app_->getMapT();
}

void StatelessObserverImpl::observeSolutionT (
  double stamp, Teuchos::ArrayRCP<const Tpetra_Vector > &nonOverlappedSolutionT,
  Teuchos::ArrayRCP<const Teuchos::Ptr<const Tpetra_Vector> >& nonOverlappedSolutionDotT)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  /*Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionT =
    app_->getOverlapSolutionT(nonOverlappedSolutionT);
  app_->getDiscretization()->writeSolutionT(*/
  //  *overlappedSolutionT, stamp, /*overlapped =*/ true);
}

} // namespace LCM
