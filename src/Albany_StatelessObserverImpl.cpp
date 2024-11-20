//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_StatelessObserverImpl.hpp"

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"

#include <string>

namespace Albany {

StatelessObserverImpl::
StatelessObserverImpl (const Teuchos::RCP<Application> &app)
  : app_(app),
  solOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Observe Solution"))
{
  force_write_solution_ = app->getForceWriteSolution(); 
}

RealType StatelessObserverImpl::
getTimeParamValueOrDefault (RealType defaultValue) const {
  const std::string label("Time");
  bool const
    use_time_param = (app_->getParamLib()->isParameter(label) == true) &&
      (app_->getSolutionMethod() != Transient);

  double const
    this_time = use_time_param == true ?
        app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>(label) :
        defaultValue;

  return this_time;
}

void StatelessObserverImpl::
observeSolution (double stamp,
                 const Teuchos::RCP<const Thyra_Vector>& x,
                 const Teuchos::RCP<const Thyra_Vector>& x_dot,
                 const Teuchos::RCP<const Thyra_Vector>& x_dotdot,
                 const Teuchos::RCP<const Thyra_MultiVector>& dxdp)
{
  Teuchos::TimeMonitor timer(*solOutTime_);
  auto overlapped_x = app_->getAdaptSolMgr()->updateAndReturnOverlapSolution(*x);
  Teuchos::RCP<Thyra_MultiVector> overlapped_dxdp;
  if (dxdp != Teuchos::null) {
    overlapped_dxdp = app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDxDp(*dxdp);
  }
  if (x_dot != Teuchos::null) {
    auto overlapped_x_dot = app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDot(*x_dot);
    if (x_dotdot != Teuchos::null) {
      auto overlapped_x_dotdot = app_->getAdaptSolMgr()->updateAndReturnOverlapSolutionDotDot(*x_dotdot);
      app_->getDiscretization()->writeSolution(
        *overlapped_x, overlapped_dxdp, *overlapped_x_dot, *overlapped_x_dotdot,
        stamp, true, force_write_solution_);
    } else {
      app_->getDiscretization()->writeSolution(
        *overlapped_x, overlapped_dxdp, *overlapped_x_dot, stamp, true, force_write_solution_);
    }
  } else {
    app_->getDiscretization()->writeSolution(*overlapped_x, overlapped_dxdp, stamp, true, force_write_solution_);
  }
}

} // namespace Albany
