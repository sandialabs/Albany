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

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
#include "PeridigmManager.hpp"
#endif
#endif

#include <string>

namespace Albany {

ObserverImpl::
ObserverImpl (const Teuchos::RCP<Application> &app)
  : StatelessObserverImpl(app)
{}

// #if defined(ALBANY_EPETRA)
// void ObserverImpl::observeSolution (
//   double stamp, const Epetra_Vector& nonOverlappedSolution,
//   const Teuchos::Ptr<const Epetra_Vector>& nonOverlappedSolutionDot)
// {
//   // If solution == "Steady" or "Continuation", we need to update the solution
//   // from the initial guess prior to writing it out, or we will not get the
//   // proper state of things like "Stress" in the Exodus file.
//   {
//     // Evaluate state field manager
//     if(nonOverlappedSolutionDot != Teuchos::null)
//       app_->evaluateStateFieldManager(stamp, nonOverlappedSolutionDot.get(),
//                                       NULL, nonOverlappedSolution);
//     else
//       app_->evaluateStateFieldManager(stamp, NULL, NULL, nonOverlappedSolution);

//     // Renames the New state as the Old state in preparation for the next step
//     app_->getStateMgr().updateStates();

// #ifdef ALBANY_PERIDIGM
// #if defined(ALBANY_EPETRA)
//     const Teuchos::RCP<LCM::PeridigmManager>&
//       peridigmManager = LCM::PeridigmManager::self();
//     if (Teuchos::nonnull(peridigmManager)) {
//       peridigmManager->writePeridigmSubModel(stamp);
//       peridigmManager->updateState();
//     }
// #endif
// #endif
//   }

//   //! update distributed parameters in the mesh
//   Teuchos::RCP<DistributedParameterLibrary> distParamLib = app_->getDistributedParameterLibrary();
//   distParamLib->scatter();
//   DistributedParameterLibrary::const_iterator it;
//   for(it = distParamLib->begin(); it != distParamLib->end(); ++it) {
//     app_->getDiscretization()->setField(*it->second->overlapped_vector(), it->second->name(),
//                                         [>overlapped<] true);
//   }

//   StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution,
//                                          nonOverlappedSolutionDot);
// }
// #endif

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

} // namespace Albany
