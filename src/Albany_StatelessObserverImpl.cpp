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
  //IKT, NOTE: solMethod == 1 corresponds to Transient
  bool const
    use_time_param = (app_->getParamLib()->isParameter(label) == true) &&
      (app_->getSolutionMethod() != 1);

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
  const Teuchos::Ptr<const Thyra_MultiVector> &nonOverlappedSolution_dxdp,
  const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDot)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_Vector> overlappedSolution =
    app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(nonOverlappedSolution);
  Teuchos::RCP<Thyra_MultiVector> overlappedSolutionDxDp = Teuchos::null; 
  if (nonOverlappedSolution_dxdp != Teuchos::null) {
    overlappedSolutionDxDp = app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDxDp(*nonOverlappedSolution_dxdp);
    /*auto out_ = Teuchos::VerboseObjectBase::getDefaultOStream(); 
    int num_param = overlappedSolutionDxDp->domain()->dim(); 
    for (int np = 0; np < num_param; np++) {
      *out_ << "\n*** StatelessObserverImpl::observeSolution overlappedSolutionDxDp" << np << " ***\n";
      Teuchos::RCP<const Thyra::VectorBase<double>> solution_dxdp_np = overlappedSolutionDxDp->col(np);
      Teuchos::Range1D range;
      RTOpPack::ConstSubVectorView<double> dxdpv;
      solution_dxdp_np->acquireDetachedView(range, &dxdpv);
      auto dxdpa = dxdpv.values();
      for (auto i = 0; i < dxdpa.size(); ++i) *out_ << dxdpa[i] << " ";
      *out_ << "\n*** StatelessObserverImpl::observeSolution overlappedSolutionDxDp" << np << " ***\n";
    }*/
  }
  if (nonOverlappedSolutionDot != Teuchos::null) {
    const Teuchos::RCP<const Thyra_Vector> overlappedSolutionDot =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(*nonOverlappedSolutionDot);
    app_->getDiscretization()->writeSolution(
      *overlappedSolution, overlappedSolutionDxDp, *overlappedSolutionDot, stamp, true);
  } 
  else {
    app_->getDiscretization()->writeSolution(*overlappedSolution, overlappedSolutionDxDp, stamp, true);
  }
}

void StatelessObserverImpl::observeSolution (
  double stamp,
  const Thyra_Vector &nonOverlappedSolution,
  const Teuchos::Ptr<const Thyra_MultiVector> &nonOverlappedSolution_dxdp,
  const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDot,
  const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDotDot)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_Vector> overlappedSolution =
    app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(nonOverlappedSolution);
  Teuchos::RCP<Thyra_MultiVector> overlappedSolutionDxDp = Teuchos::null; 
  if (nonOverlappedSolution_dxdp != Teuchos::null) {
    overlappedSolutionDxDp = app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDxDp(*nonOverlappedSolution_dxdp);
  }
  if (nonOverlappedSolutionDot != Teuchos::null) {
    const Teuchos::RCP<const Thyra_Vector> overlappedSolutionDot =
      app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(*nonOverlappedSolutionDot);
    if (nonOverlappedSolutionDotDot != Teuchos::null) {
      const Teuchos::RCP<const Thyra_Vector> overlappedSolutionDotDot =
        app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDotDot(*nonOverlappedSolutionDotDot);
      app_->getDiscretization()->writeSolution(
        *overlappedSolution, overlappedSolutionDxDp, *overlappedSolutionDot, *overlappedSolutionDotDot, 
        stamp, true);
    } 
    else {
      app_->getDiscretization()->writeSolution(
        *overlappedSolution, overlappedSolutionDxDp, *overlappedSolutionDot, stamp, true);
    }
  } 
  else {
    app_->getDiscretization()->writeSolution(
      *overlappedSolution, overlappedSolutionDxDp, stamp, true);
  }
}

void StatelessObserverImpl::observeSolution (
  double stamp, const Thyra_MultiVector &nonOverlappedSolution,
  const Teuchos::Ptr<const Thyra_MultiVector> &nonOverlappedSolution_dxdp)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  const Teuchos::RCP<const Thyra_MultiVector> overlappedSolution =
    app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionMV(nonOverlappedSolution);
  Teuchos::RCP<Thyra_MultiVector> overlappedSolutionDxDp = Teuchos::null; 
  if (nonOverlappedSolution_dxdp != Teuchos::null) {
    overlappedSolutionDxDp = app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDxDp(*nonOverlappedSolution_dxdp);
  }
  app_->getDiscretization()->writeSolutionMV(*overlappedSolution, overlappedSolutionDxDp, stamp, true);
}

} // namespace Albany
