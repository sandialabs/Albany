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

#include <Tempus_Stepper.hpp>
#include <Tempus_StepperImplicit.hpp>

namespace {
std::tuple<Teuchos::RCP<const Thyra_Vector>,
           Teuchos::RCP<const Thyra_Vector>,
           Teuchos::RCP<const Thyra_Vector>,
           Teuchos::RCP<const Thyra_MultiVector>>
unpackState (const Teuchos::RCP<const Tempus::SolutionState<ST>>& state)
{
  Teuchos::RCP<const Thyra_Vector> x       = state->getX();
  Teuchos::RCP<const Thyra_Vector> xdot    = state->getXDot();
  Teuchos::RCP<const Thyra_Vector> xdotdot = state->getXDotDot();
  Teuchos::RCP<const Thyra_MultiVector> dxdp;

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

  return std::make_tuple(x, xdot, xdotdot, dxdp);
}
}

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
observeStartTimeStep(const Tempus::Integrator<ST>& integrator)
{
  auto  history = integrator.getSolutionHistory();
  auto  state = history->getCurrentState();

  auto [x,xdot,xdotdot,dxdp] = unpackState(state.getConst());

  auto disc = app_->getDiscretization();
  auto adaptData = disc->checkForAdaptation(x,xdot,xdotdot,dxdp,is_first_time_step_);
  if (adaptData->type != AdaptationType::None) {
    disc->adapt(adaptData);
    app_->getAdaptSolMgr()->reset_solution_space(false);
    auto sol = app_->getAdaptSolMgr()->getCurrentSolution();
    auto num_time_derivs = app_->getNumTimeDerivs();

    // current state: x_old for the upcoming solve
    state->setX(sol->col(0));
    if (num_time_derivs>0) state->setXDot(sol->col(1));
    if (num_time_derivs>1) state->setXDotDot(sol->col(2));

    // working state: independent copy for NOX to mutate
    auto ws = history->getWorkingState();
    ws->setX(sol->col(0)->clone_v());
    if (num_time_derivs>0) ws->setXDot(sol->col(1)->clone_v());
    if (num_time_derivs>1) ws->setXDotDot(sol->col(2)->clone_v());

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
      auto stepper = integrator.getStepper();
      stepper->setModel(model_);
      stepper->initialize();
    }
  }
}

void PiroTempusObserver::
observeEndTimeStep(const Tempus::Integrator<ST>& integrator)
{
  auto  history = integrator.getSolutionHistory();
  auto  state   = history->getCurrentState();

  TEUCHOS_TEST_FOR_EXCEPTION (state.is_null(), std::runtime_error,
      "Error! Unexpectedly found a null current state in the tempus integrator.\n");

  // Don't observe solution if step failed to converge
  if (state->getSolutionStatus() == Tempus::Status::FAILED) {
    return;
  }

  auto time = state->getTime();
  auto [x,xdot,xdotdot,dxdp] = unpackState(state.getConst());

  if (Teuchos::nonnull(explicit_ode_model_)) {
    // Solve the algebraic constraint (e.g., the FO velocity) at the new state, so that
    // what we save to the mesh is consistent with it. The solve is cached by the model,
    // so that the first stage of the next step does not have to repeat it.
    TEUCHOS_TEST_FOR_EXCEPTION (Teuchos::nonnull(dxdp), std::logic_error,
        "Error! Explicit time integration of a DAE does not support sensitivities.\n");
    auto x_nc = state->getX();
    explicit_ode_model_->syncAlgebraicState(*x_nc,time);
    x = x_nc;
  }

  observeSolutionImpl (x,xdot,xdotdot,dxdp,time);

  // Now that a time step fully completed, we can set this to false
  is_first_time_step_ = false;
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
