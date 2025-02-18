//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PiroTempusObserver.hpp"

#include <Tempus_IntegratorForwardSensitivity.hpp>
#include <Tempus_Stepper.hpp>
#include <Tempus_StepperImplicit.hpp>

namespace Albany
{

PiroTempusObserver::
PiroTempusObserver(const Teuchos::RCP<Application>& app,
                   const Teuchos::RCP<const Thyra_ModelEvaluator>& model)
 : PiroObserver(app,model)
 , app_ (app)
{
  // Nothing else to do
}

void PiroTempusObserver::
observeEndTimeStep(const Tempus::Integrator<ST>& integrator)
{
  auto& integrator_nc = const_cast<Tempus::Integrator<ST>&>(integrator);
  auto  history_nc = integrator_nc.getNonConstSolutionHistory();
  auto  state_nc = history_nc->getCurrentState();

  TEUCHOS_TEST_FOR_EXCEPTION (state_nc.is_null(), std::runtime_error,
      "Error! Unexpectedly found a null current state in the tempus integrator.\n");

  //Don't observe solution if step failed to converge
  if (state_nc->getSolutionStatus() == Tempus::Status::FAILED) {
    return;
  }

  // In order for the DISC to decide whether to adapt or not,
  // we need to write the solution in the mesh db.
  // HOWEVER, we don't want to do a regular "observation" step,
  // since we don't want to write the solution to file, or to observe responses
  Teuchos::RCP<const Thyra_MultiVector> dxdp;
  auto integrator_ptr = Teuchos::rcpFromRef(integrator);
  auto fwd_sens_integrator = Teuchos::rcp_dynamic_cast<const Tempus::IntegratorForwardSensitivity<ST>>(integrator_ptr);
  if (Teuchos::nonnull(fwd_sens_integrator)) {
    dxdp = fwd_sens_integrator->getDxDp();
  }

  auto time     = state_nc->getTime();
  auto disc = app_->getDiscretization();
  
  auto state = state_nc.getConst();
  auto adaptData = disc->checkForAdaptation(state->getX(),state->getXDot(),state->getXDotDot(),dxdp);

  // Before observing the solution, check if we need to adapt
  if (adaptData->type!=AdaptationType::None) {
    disc->adapt (adaptData);
    // Make the solution manager import the new solution from the discretization
    app_->getAdaptSolMgr()->reset_solution_space(false);
    auto num_time_derivs = app_->getNumTimeDerivs();

    // Get new solution
    auto sol = app_->getAdaptSolMgr()->getCurrentSolution();

    // Reset vectors now that they have been adapted
    state_nc->setX(sol->col(0));
    if (num_time_derivs>0) {
      state_nc->setXDot(sol->col(1));
      if (num_time_derivs>1) {
        state_nc->setXDotDot(sol->col(2));
      }
    }
    if (Teuchos::nonnull(dxdp)) {
      dxdp = app_->getAdaptSolMgr()->getCurrentDxDp();
    }

    if (adaptData->type==AdaptationType::Topology) {
      // This should trigger the nonlinear solver to be rebuilt, which should create new linear
      // algebra objects (jac and residual)
      auto stepper = integrator.getStepper();
      stepper->setModel(model_);
      stepper->initialize();
    }
  }
  observeSolutionImpl (state->getX(),state->getXDot(),state->getXDotDot(),dxdp,time);
}

} // namespace Albany
