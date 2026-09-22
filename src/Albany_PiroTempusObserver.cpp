//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PiroTempusObserver.hpp"
#include "Albany_ExplicitODEModelEvaluator.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include <Thyra_DefaultMultiVectorProductVector.hpp>

#include <iostream>

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

  auto time     = state_nc->getTime();
  auto disc = app_->getDiscretization();
  
  auto state = state_nc.getConst();
  auto x = state->getX();
  auto xdot = state->getXDot();
  auto xdotdot = state->getXDotDot();

  // If piro created a sensitivity tempus integrator, x/xdot/xdotdot will
  // be in fact product vectors, so we need to extract the pieces
  using DMVPV = Thyra::DefaultMultiVectorProductVector<ST>;
  auto px       = Teuchos::rcp_dynamic_cast<const DMVPV>(x);
  auto pxdot    = Teuchos::rcp_dynamic_cast<const DMVPV>(xdot);
  auto pxdotdot = Teuchos::rcp_dynamic_cast<const DMVPV>(xdotdot);
  if (Teuchos::nonnull(px)) {
    x = px->getMultiVector()->col(0);
    if (Teuchos::nonnull(pxdot)) {
      xdot = pxdot->getMultiVector()->col(0);
      if (Teuchos::nonnull(pxdotdot)) {
        xdotdot = pxdotdot->getMultiVector()->col(0);
      }
    }

    const int num_param = px->getMultiVector()->domain()->dim() - 1;
    const Teuchos::Range1D rng(1, num_param);
    dxdp = px->getMultiVector()->subView(rng);
  }

  auto adaptData = disc->checkForAdaptation(x,xdot,xdotdot,dxdp);

  // Before observing the solution, check if we need to adapt
  if (adaptData->type!=AdaptationType::None) {
    disc->adapt (adaptData);
    // Make the solution manager import the new solution from the discretization
    app_->getAdaptSolMgr()->reset_solution_space(false);
    auto num_time_derivs = app_->getNumTimeDerivs();

    // Get new solution
    auto sol = app_->getAdaptSolMgr()->getCurrentSolution();

    Teuchos::RCP<Thyra_Vector> x_nc, xdot_nc, xdotdot_nc;

    // Reset vectors now that they have been adapted
    // TODO: we may need to revise these lines in case the state stores product vectors.
    x = x_nc = sol->col(0);
    state_nc->setX(x_nc);
    if (num_time_derivs>0) {
      xdot = xdot_nc = sol->col(1);
      state_nc->setXDot(xdot_nc);
      if (num_time_derivs>1) {
        xdotdot = xdotdot_nc = sol->col(2);
        state_nc->setXDotDot(xdotdot_nc);
      }
    }
    if (Teuchos::nonnull(dxdp)) {
      dxdp = app_->getAdaptSolMgr()->getCurrentDxDp();
    }

    if (Teuchos::nonnull(explicit_ode_model_)) {
      // The masks, the work vectors, the mass matrix and the algebraic solver all live
      // on the old dof layout: rebuild them for the adapted discretization.
      // NOTE: this has not been tested with an actual mesh change yet.
      Teuchos::RCP<Thyra_Vector> algebraic_mask, differential_mask;
      const bool is_dae = app_->getProblem()->getDAEMasks(*disc,algebraic_mask,differential_mask);
      TEUCHOS_TEST_FOR_EXCEPTION (!is_dae, std::logic_error,
          "Error! The problem stopped providing the DAE masks after adaptation.\n");
      explicit_ode_model_->reinitialize(algebraic_mask,differential_mask);
    }

    if (adaptData->type==AdaptationType::Topology) {
      // This should trigger the nonlinear solver to be rebuilt, which should create new linear
      // algebra objects (jac and residual)
      auto stepper = integrator.getStepper();
      stepper->setModel(model_);
      stepper->initialize();
    }
  }

  if (Teuchos::nonnull(explicit_ode_model_)) {
    // Solve the algebraic constraint (e.g., the FO velocity) at the new state, so that
    // the output is consistent. The solve is cached by the model, so that the first stage
    // of the next step does not have to repeat it.
    TEUCHOS_TEST_FOR_EXCEPTION (Teuchos::nonnull(px), std::logic_error,
        "Error! Explicit time integration of a DAE does not support sensitivities.\n");
    auto x_nc = state_nc->getX();
    explicit_ode_model_->syncAlgebraicState(*x_nc,time);
    x = x_nc;
  }

  observeSolutionImpl (x,xdot,xdotdot,dxdp,time);
}

void PiroTempusObserver::
observeEndIntegrator(const Tempus::Integrator<ST>& integrator)
{
  Tempus::IntegratorObserverBasic<ST>::observeEndIntegrator(integrator);

  if (Teuchos::nonnull(explicit_ode_model_) && app_->getComm()->getRank()==0) {
    explicit_ode_model_->printStatistics(std::cout);
  }
}

} // namespace Albany
