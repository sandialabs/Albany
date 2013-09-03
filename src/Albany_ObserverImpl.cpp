//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "AAdapt_AdaptiveSolutionManager.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Ptr.hpp"

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

Epetra_Map ObserverImpl::getNonOverlappedMap() const
{
  return *app_->getMap();
}

Teuchos::RCP<const Tpetra_Map> ObserverImpl::getNonOverlappedMapT() const
{
  return app_->getMapT();
}

void ObserverImpl::observeSolution(
    double stamp,
    const Epetra_Vector &nonOverlappedSolution,
    Teuchos::Ptr<const Epetra_Vector> nonOverlappedSolutionDot)
{
  // If solution == "Steady" or "Continuation", we need to update the solution from the initial guess prior to
  // writing it out, or we will not get the proper state of things like "Stress" in the Exodus file.
  {
    // Evaluate state field manager
    app_->evaluateStateFieldManager(stamp, nonOverlappedSolutionDot.get(), nonOverlappedSolution);

    // Renames the New state as the Old state in preparation for the next step
    app_->getStateMgr().updateStates();
  }

  Teuchos::TimeMonitor timer(*solOutTime_);
  {
    const Teuchos::Ptr<const Epetra_Vector> overlappedSolution(
        app_->getAdaptSolMgr()->getOverlapSolution(nonOverlappedSolution));
    app_->getDiscretization()->writeSolution(*overlappedSolution, stamp, /*overlapped =*/ true);
  }
}

void ObserverImpl::observeSolutionT(
    double stamp,
    const Tpetra_Vector &nonOverlappedSolutionT,
    Teuchos::Ptr<const Tpetra_Vector> nonOverlappedSolutionDotT)
{
  // If solution == "Steady" or "Continuation", we need to update the solution from the initial guess prior to
  // writing it out, or we will not get the proper state of things like "Stress" in the Exodus file.
  {
    // Evaluate state field manager
    app_->evaluateStateFieldManagerT(stamp, nonOverlappedSolutionDotT, nonOverlappedSolutionT);

    // Renames the New state as the Old state in preparation for the next step
    app_->getStateMgr().updateStates();
  }

  Teuchos::TimeMonitor timer(*solOutTime_);
  {
    const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionT =
      app_->getOverlapSolutionT(nonOverlappedSolutionT);
    app_->getDiscretization()->writeSolutionT(*overlappedSolutionT, stamp, /*overlapped =*/ true);
  }
}

} // namespace Albany
