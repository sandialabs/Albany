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
ObserverImpl (const Teuchos::RCP<Albany::Application> &app, 
              Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > &apps) :
  StatelessObserverImpl(app, apps) 
{
  //FIXME: remove app as argument 
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
}


void ObserverImpl::observeSolutionT(
  double stamp, const Tpetra_Vector &nonOverlappedSolutionT,
  const Teuchos::Ptr<const Tpetra_Vector>& nonOverlappedSolutionDotT)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  app_->evaluateStateFieldManagerT(stamp, nonOverlappedSolutionDotT,
                                   Teuchos::null, nonOverlappedSolutionT);
  app_->getStateMgr().updateStates();

  StatelessObserverImpl::observeSolutionT(stamp, nonOverlappedSolutionT,
                                          nonOverlappedSolutionDotT);
}

} // namespace LCM

