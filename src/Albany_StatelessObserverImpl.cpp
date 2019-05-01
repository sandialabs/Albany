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

Teuchos::RCP<const Thyra_VectorSpace>
StatelessObserverImpl::getNonOverlappedVectorSpace () const {
  return app_->getVectorSpace();
}

void StatelessObserverImpl::observeSolution (
  double stamp,
  const Thyra_Vector &nonOverlappedSolution,
  const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDot)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_Vector> overlappedSolution =
    app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(nonOverlappedSolution);
  if (nonOverlappedSolutionDot != Teuchos::null) {
    const Teuchos::RCP<const Thyra_Vector> overlappedSolutionDot =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(*nonOverlappedSolutionDot);
    app_->getDiscretization()->writeSolution(
      *overlappedSolution, *overlappedSolutionDot, stamp, /*overlapped =*/ true);
  } else {
    app_->getDiscretization()->writeSolution(
      *overlappedSolution, stamp, /*overlapped =*/ true);
  }
}

void StatelessObserverImpl::observeSolution (
  double stamp,
  const Thyra_Vector &nonOverlappedSolution,
  const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDot,
  const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDotDot)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_Vector> overlappedSolution =
    app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(nonOverlappedSolution);
  if (nonOverlappedSolutionDot != Teuchos::null) {
    const Teuchos::RCP<const Thyra_Vector> overlappedSolutionDot =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(*nonOverlappedSolutionDot);
    if (nonOverlappedSolutionDotDot != Teuchos::null) {
      const Teuchos::RCP<const Thyra_Vector> overlappedSolutionDotDot =
        app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDotDot(*nonOverlappedSolutionDotDot);
      app_->getDiscretization()->writeSolution(
        *overlappedSolution, *overlappedSolutionDot, *overlappedSolutionDotDot, 
        stamp, /*overlapped =*/ true);
    } else {
      app_->getDiscretization()->writeSolution(
        *overlappedSolution, *overlappedSolutionDot, stamp, /*overlapped =*/ true);
    }
  } else {
    app_->getDiscretization()->writeSolution(
      *overlappedSolution, stamp, /*overlapped =*/ true);
  }
}

void StatelessObserverImpl::observeSolution (
  double stamp, const Thyra_MultiVector &nonOverlappedSolution)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_MultiVector> overlappedSolution =
    app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionMV(nonOverlappedSolution);
  app_->getDiscretization()->writeSolutionMV(
    *overlappedSolution, stamp, /*overlapped =*/ true);
}

} // namespace Albany
