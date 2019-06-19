//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ObserverImpl.hpp"

#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#if defined(ALBANY_EPETRA)
# include "AAdapt_AdaptiveSolutionManager.hpp"
#endif

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Ptr.hpp"

#include <string>

namespace Albany {

ObserverImpl::
ObserverImpl (const Teuchos::RCP<Application> &app)
  : StatelessObserverImpl(app)
{}

void ObserverImpl::
observeSolution(double stamp,
                const Thyra_Vector& nonOverlappedSolution,
                const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDot,
                const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDotDot)
{
  app_->evaluateStateFieldManager (stamp,
                                   nonOverlappedSolution,
                                   nonOverlappedSolutionDot,
                                   nonOverlappedSolutionDotDot);

  app_->getStateMgr().updateStates();

  StatelessObserverImpl::observeSolution (stamp,
                                          nonOverlappedSolution,
                                          nonOverlappedSolutionDot,
                                          nonOverlappedSolutionDotDot);
}

void ObserverImpl::
observeSolution(double stamp,
                const Thyra_MultiVector& nonOverlappedSolution)
{
  app_->evaluateStateFieldManager(stamp, nonOverlappedSolution);
  app_->getStateMgr().updateStates();
  StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution);
}

void ObserverImpl::
observeParameter(const std::string& param)
{
  app_->getPhxSetup()->init_unsaved_param(param);
}

} // namespace Albany
