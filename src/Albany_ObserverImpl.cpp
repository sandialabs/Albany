//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"
#ifdef ALBANY_EPETRA
#include "AAdapt_AdaptiveSolutionManager.hpp"
#endif

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Ptr.hpp"

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
#include "PeridigmManager.hpp"
#endif
#endif

#include <string>

namespace Albany {

ObserverImpl::ObserverImpl(const Teuchos::RCP<Application> &app) :
   app_(app),
   solOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Output to File"))
{
  // Nothing to do
}

RealType ObserverImpl::getTimeParamValueOrDefault(RealType defaultValue) const
{
  const std::string label("Time");

  return (app_->getParamLib()->isParameter(label)) ?
    app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(label) :
    defaultValue;
}

#ifdef ALBANY_EPETRA
Epetra_Map ObserverImpl::getNonOverlappedMap() const
{
  return *app_->getMap();
}
#endif

Teuchos::RCP<const Tpetra_Map> ObserverImpl::getNonOverlappedMapT() const
{
  return app_->getMapT();
}

#ifdef ALBANY_EPETRA
void ObserverImpl::observeSolution(
    double stamp,
    const Epetra_Vector &nonOverlappedSolution,
    Teuchos::Ptr<const Epetra_Vector> nonOverlappedSolutionDot)
{
  // If solution == "Steady" or "Continuation", we need to update the solution from the initial guess prior to
  // writing it out, or we will not get the proper state of things like "Stress" in the Exodus file.
  {
    // Evaluate state field manager
    if(nonOverlappedSolutionDot != Teuchos::null)
      app_->evaluateStateFieldManager(stamp, nonOverlappedSolutionDot.get(), NULL, nonOverlappedSolution);
    else
      app_->evaluateStateFieldManager(stamp, NULL, NULL, nonOverlappedSolution);

    // Renames the New state as the Old state in preparation for the next step
    app_->getStateMgr().updateStates();

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
    LCM::PeridigmManager& peridigmManager = LCM::PeridigmManager::self();
    peridigmManager.writePeridigmSubModel(stamp);
    peridigmManager.updateState();
#endif
#endif
  }

  Teuchos::TimeMonitor timer(*solOutTime_);
  {
    const Teuchos::Ptr<const Epetra_Vector> overlappedSolution(
        app_->getAdaptSolMgr()->getOverlapSolution(nonOverlappedSolution));
    app_->getDiscretization()->writeSolution(*overlappedSolution, stamp, /*overlapped =*/ true);
  }
}
#endif

void ObserverImpl::observeSolutionT(
    double stamp,
    const Tpetra_Vector &nonOverlappedSolutionT,
    Teuchos::Ptr<const Tpetra_Vector> nonOverlappedSolutionDotT)
{
  // If solution == "Steady" or "Continuation", we need to update the solution from the initial guess prior to
  // writing it out, or we will not get the proper state of things like "Stress" in the Exodus file.
  {
    // Evaluate state field manager
    app_->evaluateStateFieldManagerT(stamp, nonOverlappedSolutionDotT, Teuchos::null, nonOverlappedSolutionT);

    if ( ! app_->isModelEvaluatorTCallingWriteSolutionT()) {
      //exo-hack As part of the hack, do not call updateStates until after the
      // RF are called. This breaks I think two tests because of a change in
      // behavior of the LOCA start step's predictor. In any case, this hack is
      // active only when an IP-to-nodal RF is used.

      // Renames the New state as the Old state in preparation for the next step
      app_->getStateMgr().updateStates();
    }
  }

  Teuchos::TimeMonitor timer(*solOutTime_);
  {
    const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionT =
      app_->getOverlapSolutionT(nonOverlappedSolutionT);
    app_->getDiscretization()->writeSolutionToMeshDatabaseT(
      *overlappedSolutionT, stamp, /*overlapped =*/ true);
    if ( ! app_->isModelEvaluatorTCallingWriteSolutionT())
      app_->getDiscretization()->writeSolutionToFileT(
        *overlappedSolutionT, stamp, /*overlapped =*/ true);
  }
}

} // namespace Albany
