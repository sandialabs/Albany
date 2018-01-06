//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_StatelessObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"
#if defined(ALBANY_EPETRA)
#include "AAdapt_AdaptiveSolutionManager.hpp"
#endif

#include "Teuchos_TimeMonitor.hpp"

#include <string>

namespace Albany {

StatelessObserverImpl::
StatelessObserverImpl (const Teuchos::RCP<Application> &app)
  : app_(app),
  solOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Output to File"))
{}

RealType StatelessObserverImpl::
getTimeParamValueOrDefault (RealType defaultValue) const {
  const std::string label("Time");
  //IKT, NOTE: solMethod == 2 corresponds to TransientTempus
#ifdef ALBANY_LCM
  bool const
    use_time_param = (app_->getParamLib()->isParameter(label) == true) &&
      (app_->getSchwarzAlternating() == false) && (app_->getSolutionMethod() != 2);
#else
  bool const
    use_time_param = (app_->getParamLib()->isParameter(label) == true) &&
      (app_->getSolutionMethod() != 2);
#endif

  double const
    this_time = use_time_param == true ?
        app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(label) :
        defaultValue;

  return this_time;

}

#if defined(ALBANY_EPETRA)
const Epetra_Map& StatelessObserverImpl::getNonOverlappedMap () const {
  return *app_->getMap();
}
#endif

Teuchos::RCP<const Tpetra_Map>
StatelessObserverImpl::getNonOverlappedMapT () const {
  return app_->getMapT();
}

#if defined(ALBANY_EPETRA)
void StatelessObserverImpl::observeSolution (
  double stamp, const Epetra_Vector &nonOverlappedSolution,
  const Teuchos::Ptr<const Epetra_Vector>& nonOverlappedSolutionDot)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::Ptr<const Epetra_Vector> overlappedSolution(
    app_->getAdaptSolMgr()->getOverlapSolution(nonOverlappedSolution));
  
  if (nonOverlappedSolutionDot.is_null()) {
    app_->getDiscretization()->writeSolution(*overlappedSolution, stamp,
                                             /*overlapped =*/ true);
  }
  else {  
    const Teuchos::Ptr<const Epetra_Vector> overlappedSolutionDot(
      app_->getAdaptSolMgr()->getOverlapSolution(*nonOverlappedSolutionDot));
    app_->getDiscretization()->writeSolution(*overlappedSolution, *overlappedSolutionDot, stamp,
                                             /*overlapped =*/ true);
  }
}
#endif

void StatelessObserverImpl::observeSolutionT (
  double stamp, const Tpetra_Vector &nonOverlappedSolutionT,
  const Teuchos::Ptr<const Tpetra_Vector>& nonOverlappedSolutionDotT)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionT =
    app_->getAdaptSolMgrT()->updateAndReturnOverlapSolutionT(nonOverlappedSolutionT);
  if (nonOverlappedSolutionDotT != Teuchos::null) {
    const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionDotT =
      app_->getAdaptSolMgrT()->updateAndReturnOverlapSolutionDotT(*nonOverlappedSolutionDotT);
    app_->getDiscretization()->writeSolutionT(
      *overlappedSolutionT, *overlappedSolutionDotT, stamp, /*overlapped =*/ true);
  }
  else {
    app_->getDiscretization()->writeSolutionT(
      *overlappedSolutionT, stamp, /*overlapped =*/ true);
  }
}

void StatelessObserverImpl::observeSolutionT (
  double stamp, const Tpetra_Vector &nonOverlappedSolutionT,
  const Teuchos::Ptr<const Tpetra_Vector>& nonOverlappedSolutionDotT,
  const Teuchos::Ptr<const Tpetra_Vector>& nonOverlappedSolutionDotDotT)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionT =
    app_->getAdaptSolMgrT()->updateAndReturnOverlapSolutionT(nonOverlappedSolutionT);
  if (nonOverlappedSolutionDotT != Teuchos::null) {
    const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionDotT =
      app_->getAdaptSolMgrT()->updateAndReturnOverlapSolutionDotT(*nonOverlappedSolutionDotT);
    if (nonOverlappedSolutionDotDotT != Teuchos::null) {
      const Teuchos::RCP<const Tpetra_Vector> overlappedSolutionDotDotT =
        app_->getAdaptSolMgrT()->updateAndReturnOverlapSolutionDotDotT(*nonOverlappedSolutionDotDotT);
      app_->getDiscretization()->writeSolutionT(
        *overlappedSolutionT, *overlappedSolutionDotT, *overlappedSolutionDotDotT, 
        stamp, /*overlapped =*/ true);
    }
    else {
      app_->getDiscretization()->writeSolutionT(
        *overlappedSolutionT, *overlappedSolutionDotT, stamp, /*overlapped =*/ true);
   }
  }
  else {
    app_->getDiscretization()->writeSolutionT(
      *overlappedSolutionT, stamp, /*overlapped =*/ true);
  }
}


void StatelessObserverImpl::observeSolutionT (
  double stamp, const Tpetra_MultiVector &nonOverlappedSolutionT)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Tpetra_MultiVector> overlappedSolutionT =
    app_->getAdaptSolMgrT()->updateAndReturnOverlapSolutionMV(nonOverlappedSolutionT);
  app_->getDiscretization()->writeSolutionMV(
    *overlappedSolutionT, stamp, /*overlapped =*/ true);
}

} // namespace Albany
