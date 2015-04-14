//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_ObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Ptr.hpp"

#include <string>

namespace LCM {

ObserverImpl::
ObserverImpl (Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > &apps) :
  StatelessObserverImpl(apps) 
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
}


void ObserverImpl::observeSolutionT(
  double stamp, Teuchos::ArrayRCP<const Tpetra_Vector > &nonOverlappedSolutionT,
  Teuchos::ArrayRCP<const Teuchos::Ptr<const Tpetra_Vector> >& nonOverlappedSolutionDotT)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
/*
  app_->evaluateStateFieldManagerT(stamp, nonOverlappedSolutionDotT,
                                   Teuchos::null, nonOverlappedSolutionT);
  app_->getStateMgr().updateStates();

  StatelessObserverImpl::observeSolutionT(stamp, nonOverlappedSolutionT,
                                          nonOverlappedSolutionDotT);
*/
}

} // namespace LCM

