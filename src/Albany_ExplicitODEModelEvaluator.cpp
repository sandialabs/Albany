//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ExplicitODEModelEvaluator.hpp"

#include "Albany_ThyraUtils.hpp"

#include "Thyra_NonlinearSolver_NOX.hpp"
#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_LinearOpWithSolveBase.hpp"
#include "Thyra_VectorStdOps.hpp"

#include "Tempus_StepperFactory.hpp"
#include "Tempus_Stepper.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"

#include <algorithm>
#include <cmath>

namespace Albany {

// ======================= Tempus stepper inspection ======================= //

bool usesExplicitTempusStepper (const Teuchos::ParameterList& piroTempusParams)
{
  // Work on a copy: we must not add defaults to the input list
  Teuchos::ParameterList tempus_pl(piroTempusParams);

  // Same layout Piro::TempusSolver uses to locate the stepper list
  const auto integrator_name = tempus_pl.get<std::string>("Integrator Name","Tempus Integrator");
  if (!tempus_pl.isSublist(integrator_name)) {
    return false;
  }
  const auto stepper_name = tempus_pl.sublist(integrator_name).get<std::string>("Stepper Name","Tempus Stepper");
  if (!tempus_pl.isSublist(stepper_name)) {
    return false;
  }
  auto stepper_pl = Teuchos::rcp(new Teuchos::ParameterList(tempus_pl.sublist(stepper_name)));
  const auto stepper_type = stepper_pl->get<std::string>("Stepper Type","Backward Euler");

  // Steppers with sub-steppers: the sub-stepper decides, and it may be implicit
  if (stepper_type=="Operator Split" || stepper_type=="Subcycling") {
    return false;
  }

  // Let Tempus classify its own stepper (no model: we only query the stepper type)
  Teuchos::RCP<Tempus::Stepper<ST>> stepper;
  try {
    Tempus::StepperFactory<ST> factory;
    stepper = factory.createStepper(stepper_pl);
  } catch (...) {
    // Unrecognized stepper type: Piro/Tempus will report it
    return false;
  }

  return stepper->isExplicit() && !stepper->isImplicit() &&
         stepper->getOrderODE()==Tempus::FIRST_ORDER_ODE;
}

// ======================= ExplicitODEModelEvaluator ======================= //

ExplicitODEModelEvaluator::
ExplicitODEModelEvaluator (const Teuchos::RCP<Thyra_ModelEvaluator>&   model,
                           const Teuchos::RCP<const Thyra_Vector>&     algebraic_mask,
                           const Teuchos::RCP<const Thyra_Vector>&     differential_mask,
                           const Teuchos::RCP<Teuchos::ParameterList>& params,
                           const Teuchos::RCP<Teuchos::ParameterList>& noxParams,
                           const Teuchos::RCP<Thyra_LOWS_Factory>&     algebraicLowsFactory,
                           const Teuchos::RCP<Thyra_LOWS_Factory>&     massLowsFactory)
 : Thyra::ModelEvaluatorDelegatorBase<ST>(model)
 , alg_lows_factory_(algebraicLowsFactory)
 , mass_lows_factory_(massLowsFactory)
{
  using MEB = Thyra::ModelEvaluatorBase;

  // ---- Checks on the underlying model ---- //
  {
    const auto in  = model->createInArgs();
    const auto out = model->createOutArgs();
    const bool ok = in.supports(MEB::IN_ARG_x) && in.supports(MEB::IN_ARG_x_dot) &&
                    in.supports(MEB::IN_ARG_alpha) && in.supports(MEB::IN_ARG_beta) &&
                    out.supports(MEB::OUT_ARG_f) && out.supports(MEB::OUT_ARG_W_op);
    TEUCHOS_TEST_FOR_EXCEPTION (!ok, std::invalid_argument,
        "Error! ExplicitODEModelEvaluator requires an implicit model supporting\n"
        "       InArgs x, x_dot, alpha, beta and OutArgs f, W_op.\n");
  }

  // ---- Options ---- //
  Teuchos::ParameterList pl;
  if (Teuchos::nonnull(params)) {
    pl = *params;
  }
  pl.validateParametersAndSetDefaults(*getValidParameters(),0);

  const auto update = pl.get<std::string>("Algebraic Update");
  TEUCHOS_TEST_FOR_EXCEPTION (update!="Every Stage" && update!="Once Per Step", std::invalid_argument,
      "Error! Invalid 'Algebraic Update' value '" << update << "'. Valid: 'Every Stage', 'Once Per Step'.\n");
  lagged_ = (update=="Once Per Step");

  lumped_        = pl.get<bool>("Lump Mass Matrix");
  constant_mass_ = pl.get<bool>("Constant Mass Matrix");
  TEUCHOS_TEST_FOR_EXCEPTION (!lumped_ && mass_lows_factory_.is_null(), std::invalid_argument,
      "Error! ExplicitODEModelEvaluator: a consistent mass matrix requires a mass LOWS factory.\n");

  // NOX modifies its parameter list (status tests, defaults), so keep our own copy
  nox_params_ = Teuchos::rcp(new Teuchos::ParameterList("NOX"));
  if (Teuchos::nonnull(noxParams)) {
    *nox_params_ = *noxParams;
  }

  setMasks(algebraic_mask,differential_mask);
  allocateWorkVectors();

  // ---- Algebraic solve: Newton on the constraint, with the other dofs fixed ---- //
  constraint_model_ = Teuchos::rcp(new AlgebraicConstraintModelEvaluator(model,alg_mask_));
  createAlgebraicSolver();
}

ExplicitODEModelEvaluator::~ExplicitODEModelEvaluator () = default;

Teuchos::RCP<const Teuchos::ParameterList>
ExplicitODEModelEvaluator::getValidParameters ()
{
  auto pl = Teuchos::rcp(new Teuchos::ParameterList("Valid Explicit Time Integration Parameters"));
  pl->set<std::string>("Algebraic Update", "Every Stage",
      "When to solve the algebraic constraint: 'Every Stage' or 'Once Per Step' (lagged)");
  pl->set<bool>("Lump Mass Matrix", false,
      "Whether to lump the mass matrix of the differential dofs (same option as in Piro->Tempus)");
  pl->set<bool>("Constant Mass Matrix", false,
      "Whether the mass matrix is computed only once (same option as in Piro->Tempus)");
  return pl;
}

// ------------------------------- Setup ------------------------------- //

void ExplicitODEModelEvaluator::
setMasks (const Teuchos::RCP<const Thyra_Vector>& algebraic_mask,
          const Teuchos::RCP<const Thyra_Vector>& differential_mask)
{
  const auto x_vs = this->getUnderlyingModel()->get_x_space();

  TEUCHOS_TEST_FOR_EXCEPTION (algebraic_mask.is_null() || differential_mask.is_null(), std::invalid_argument,
      "Error! ExplicitODEModelEvaluator: null algebraic/differential mask.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (!algebraic_mask->space()->isCompatible(*x_vs) ||
                              !differential_mask->space()->isCompatible(*x_vs), std::invalid_argument,
      "Error! ExplicitODEModelEvaluator: masks must be in the solution vector space.\n");

  // Masks are stored as 0/1 vectors
  alg_mask_      = binaryMask(*algebraic_mask);
  diff_mask_     = binaryMask(*differential_mask);
  non_diff_mask_ = binaryMask(*differential_mask,/* complement = */ true);

  {
    // Sanity checks: disjoint masks, and at least one differential dof
    auto overlap = Thyra::createMember(x_vs);
    overlap->assign(0.0);
    Thyra::ele_wise_prod(1.0,*alg_mask_,*diff_mask_,overlap.ptr());
    TEUCHOS_TEST_FOR_EXCEPTION (Thyra::sum(*overlap)!=0.0, std::invalid_argument,
        "Error! ExplicitODEModelEvaluator: some dofs are flagged as both algebraic and differential.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (Thyra::sum(*diff_mask_)==0.0, std::invalid_argument,
        "Error! ExplicitODEModelEvaluator: no differential dofs.\n");
  }

  if (!lumped_) {
    if (Teuchos::nonnull(mass_eliminator_)) {
      mass_eliminator_->setMask(non_diff_mask_);
    } else {
      mass_eliminator_ = Teuchos::rcp(new MaskedDofsEliminator(non_diff_mask_));
    }
  }
}

void ExplicitODEModelEvaluator::allocateWorkVectors ()
{
  const auto model = this->getUnderlyingModel();
  const auto x_vs  = model->get_x_space();

  x_work_     = Thyra::createMember(x_vs);
  x_sync_     = Thyra::createMember(x_vs);
  x_scratch_  = Thyra::createMember(x_vs);
  x_dot_zero_ = Thyra::createMember(x_vs);
  mass_sol_   = Thyra::createMember(x_vs);
  cache_key_  = Thyra::createMember(x_vs);
  cache_x_    = Thyra::createMember(x_vs);
  residual_   = Thyra::createMember(model->get_f_space());
  x_dot_zero_->assign(0.0);
}

void ExplicitODEModelEvaluator::createAlgebraicSolver ()
{
  auto constraint_with_solve = Teuchos::rcp(
      new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(constraint_model_,alg_lows_factory_));

  // NOX modifies its parameter list (status tests, defaults), so give it a copy
  auto nox_pl = Teuchos::rcp(new Teuchos::ParameterList(*nox_params_));
  nox_solver_ = Teuchos::rcp(new Thyra::NOXNonlinearSolver());
  nox_solver_->setParameterList(nox_pl);
  nox_solver_->setModel(constraint_with_solve);
}

void ExplicitODEModelEvaluator::
reinitialize (const Teuchos::RCP<const Thyra_Vector>& algebraic_mask,
              const Teuchos::RCP<const Thyra_Vector>& differential_mask)
{
  setMasks(algebraic_mask,differential_mask);
  allocateWorkVectors();

  // The constraint model and the Newton solver hold vectors and linear algebra
  // objects built on the old solution space
  constraint_model_->setMask(alg_mask_);
  createAlgebraicSolver();

  // So do the mass matrix and the cached algebraic solve
  mass_op_         = Teuchos::null;
  mass_lows_       = Teuchos::null;
  inv_lumped_mass_ = Teuchos::null;
  mass_computed_   = false;
  cache_valid_     = false;
}

// ------------------------- Thyra ME interface ------------------------- //

Thyra_InArgs ExplicitODEModelEvaluator::createInArgs () const
{
  using MEB = Thyra::ModelEvaluatorBase;
  const auto model = this->getUnderlyingModel();
  MEB::InArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());
  result.set_Np_Ng(model->Np(),model->Ng());
  result.setSupports(MEB::IN_ARG_x, true);
  result.setSupports(MEB::IN_ARG_t, true);
  result.setSupports(MEB::IN_ARG_step_size, true);
  return static_cast<Thyra_InArgs>(result);
}

Thyra_InArgs ExplicitODEModelEvaluator::getNominalValues () const
{
  auto nominal = createInArgs();
  nominal.setArgs(this->getUnderlyingModel()->getNominalValues(), /* ignoreUnsupported = */ true);
  return nominal;
}

Thyra_InArgs ExplicitODEModelEvaluator::getLowerBounds () const
{
  auto bounds = createInArgs();
  bounds.setArgs(this->getUnderlyingModel()->getLowerBounds(), /* ignoreUnsupported = */ true);
  return bounds;
}

Thyra_InArgs ExplicitODEModelEvaluator::getUpperBounds () const
{
  auto bounds = createInArgs();
  bounds.setArgs(this->getUnderlyingModel()->getUpperBounds(), /* ignoreUnsupported = */ true);
  return bounds;
}

Teuchos::RCP<Thyra_LOWS> ExplicitODEModelEvaluator::create_W () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
      "Error! ExplicitODEModelEvaluator is an explicit ODE model: it does not provide W.\n"
      "       It can only be used with an explicit Tempus stepper, and Piro must not wrap it\n"
      "       in an InvertMassMatrixDecorator ('Invert Mass Matrix: false' in the Tempus list).\n");
  return Teuchos::null;
}

Thyra_OutArgs ExplicitODEModelEvaluator::createOutArgsImpl () const
{
  using MEB = Thyra::ModelEvaluatorBase;
  const auto model = this->getUnderlyingModel();
  MEB::OutArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());
  result.set_Np_Ng(model->Np(),model->Ng());
  result.setSupports(MEB::OUT_ARG_f, true);
  return static_cast<Thyra_OutArgs>(result);
}

void ExplicitODEModelEvaluator::
evalModelImpl (const Thyra_InArgs&  inArgs,
               const Thyra_OutArgs& outArgs) const
{
  const auto model = this->getUnderlyingModel();
  const auto x = inArgs.get_x();
  TEUCHOS_TEST_FOR_EXCEPTION (x.is_null(), std::logic_error,
      "Error! ExplicitODEModelEvaluator::evalModel called with a null x.\n");
  const ST t  = inArgs.get_t();
  const ST dt = inArgs.get_step_size();

  const auto f = outArgs.get_f();
  bool need_g = false;
  for (int j=0; j<outArgs.Ng(); ++j) {
    need_g = need_g || Teuchos::nonnull(outArgs.get_g(j));
  }
  if (f.is_null() && !need_g) {
    return;
  }

  // 1. Complete x with algebraic dofs consistent with its other dofs
  Thyra::copy(*x,x_work_.ptr());
  if (lagged_ && cache_valid_) {
    // Lagged: reuse the algebraic dofs of the last consistent state
    // (updated only in syncAlgebraicState, i.e., once per step)
    copyFromCache(*x_work_,/* also_prescribed = */ true);
  } else {
    makeConsistent(*x_work_,t,dt);
  }

  // 2. Residual (with x_dot=0) and responses at the completed state.
  //    Note: the model may overwrite entries of x_work_ in place (SDBCs).
  const auto modelInArgs = underlyingInArgs(x_work_,t,dt,0.0,1.0,&inArgs);
  auto modelOutArgs = model->createOutArgs();
  if (Teuchos::nonnull(f)) {
    modelOutArgs.set_f(residual_);
  }
  for (int j=0; j<outArgs.Ng(); ++j) {
    modelOutArgs.set_g(j,outArgs.get_g(j));
  }
  model->evalModel(modelInArgs,modelOutArgs);

  // 3. x_dot = -M^{-1} r, on the differential dofs only
  if (Teuchos::nonnull(f)) {
    if (!mass_computed_ || !constant_mass_) {
      computeMassMatrix(*x_work_,t,dt);
    }
    Thyra::ele_wise_scale(*diff_mask_,residual_.ptr());
    applyInverseMass(*residual_,*f);
    ++num_rhs_evals_;
  }
}

// --------------------------- Algebraic solve --------------------------- //

void ExplicitODEModelEvaluator::
syncAlgebraicState (Thyra_Vector& x, const ST t) const
{
  Thyra::copy(x,x_sync_.ptr());
  makeConsistent(*x_sync_,t,last_dt_);
  Thyra::copy(*x_sync_,Teuchos::ptrFromRef(x));
}

Teuchos::RCP<const Thyra_Vector>
ExplicitODEModelEvaluator::getLastConsistentState () const
{
  return cache_valid_ ? cache_x_.getConst() : Teuchos::null;
}

void ExplicitODEModelEvaluator::
makeConsistent (Thyra_Vector& x, const ST t, const ST dt) const
{
  if (dt!=0.0) {
    last_dt_ = dt;
  }

  if (matchesCache(x,t)) {
    copyFromCache(x,/* also_prescribed = */ true);
    ++num_cache_hits_;
    return;
  }

  // Remember the state as given (used as cache key)
  Thyra::copy(x,cache_key_.ptr());

  constraint_model_->setFixedValues(x);
  constraint_model_->setTime(t);
  constraint_model_->setStepSize(last_dt_);

  // Warm start from the last consistent algebraic dofs (algebraic dofs only: the
  // other dofs of x are the ones we are solving for)
  if (cache_valid_) {
    copyFromCache(x,/* also_prescribed = */ false);
  }

  const auto status = nox_solver_->solve(&x);
  ++num_solves_;
  if (Teuchos::nonnull(status.extraParameters) &&
      status.extraParameters->isType<int>("Number of Iterations")) {
    num_newton_its_ += status.extraParameters->get<int>("Number of Iterations");
  }

  // An explicit Tempus stepper has no way to see a failed evaluation, so there is
  // nothing to gain from continuing with an inconsistent state
  TEUCHOS_TEST_FOR_EXCEPTION (status.solveStatus!=Thyra::SOLVE_STATUS_CONVERGED, std::runtime_error,
      "Error! ExplicitODEModelEvaluator: the algebraic solve did not converge (t = " << t << ").\n"
      "       Reduce the time step, or revisit the nonlinear solver settings (Piro->NOX).\n");

  // x now holds the consistent state (incl. values prescribed in place by the model)
  Thyra::copy(x,cache_x_.ptr());
  cache_t_     = t;
  cache_valid_ = true;
}

bool ExplicitODEModelEvaluator::
matchesCache (const Thyra_Vector& x, const ST t) const
{
  if (!cache_valid_ || t!=cache_t_) {
    return false;
  }

  // Compare the non-algebraic dofs with both the state as given in the last solve,
  // and the consistent state that came out of it (they differ only at dofs that
  // the model prescribes in place, e.g. SDBCs).
  auto x_data   = getLocalData(x);
  auto key_data = getLocalData(*cache_key_);
  auto c_data   = getLocalData(*cache_x_);
  auto a_data   = getLocalData(*alg_mask_);
  ST local[2] = {0.0, 0.0};
  const LO n = x_data.size();
  for (LO i=0; i<n; ++i) {
    if (a_data[i]!=0.0) continue;
    local[0] = std::max(local[0],std::abs(x_data[i]-key_data[i]));
    local[1] = std::max(local[1],std::abs(x_data[i]-c_data[i]));
  }
  ST global[2];
  Teuchos::reduceAll(*getComm(x.space()),Teuchos::REDUCE_MAX,2,local,global);
  return std::min(global[0],global[1])==0.0;
}

void ExplicitODEModelEvaluator::
copyFromCache (Thyra_Vector& x, const bool also_prescribed) const
{
  // Copy the algebraic dofs of the last consistent state and, if requested, the dofs
  // that the model prescribed in place during the last solve (where the consistent
  // state differs from the state given to that solve, e.g. SDBC values).
  auto x_data   = getNonconstLocalData(x);
  auto key_data = getLocalData(*cache_key_);
  auto c_data   = getLocalData(*cache_x_);
  auto a_data   = getLocalData(*alg_mask_);
  const LO n = x_data.size();
  for (LO i=0; i<n; ++i) {
    if (a_data[i]!=0.0 || (also_prescribed && key_data[i]!=c_data[i])) {
      x_data[i] = c_data[i];
    }
  }
}

Thyra_InArgs ExplicitODEModelEvaluator::
underlyingInArgs (const Teuchos::RCP<const Thyra_Vector>& x,
                  const ST t, const ST dt,
                  const ST alpha, const ST beta,
                  const Thyra_InArgs* inArgs) const
{
  using MEB = Thyra::ModelEvaluatorBase;
  const auto model = this->getUnderlyingModel();
  auto in = model->createInArgs();
  in.set_x(x);
  if (inArgs!=nullptr) {
    for (int l=0; l<in.Np(); ++l) {
      in.set_p(l,inArgs->get_p(l));
    }
  }
  // Always a non-null x_dot: with a null x_dot Albany drops t, dt, alpha and beta
  in.set_x_dot(x_dot_zero_);
  in.set_alpha(alpha);
  in.set_beta(beta);
  if (in.supports(MEB::IN_ARG_t))                 { in.set_t(t); }
  if (in.supports(MEB::IN_ARG_step_size))         { in.set_step_size(dt); }
  if (in.supports(MEB::IN_ARG_W_x_dot_dot_coeff)) { in.set_W_x_dot_dot_coeff(0.0); }
  return in;
}

// ----------------------------- Mass matrix ----------------------------- //

void ExplicitODEModelEvaluator::
computeMassMatrix (const Thyra_Vector& x, const ST t, const ST dt) const
{
  const auto model = this->getUnderlyingModel();

  if (lumped_) {
    // Row sums of M, on the differential dofs only. Since f is affine in x_dot, M*1 =
    // f(x_dot=1,x,t) - f(x_dot=0,x,t). Unlike the row sums of W(alpha=1,beta=0), this
    // is not affected by the elimination of the columns of strong Dirichlet dofs.
    auto x_dot_ones = Thyra::createMember(x.space());
    x_dot_ones->assign(1.0);
    auto row_sums = Thyra::createMember(model->get_f_space());
    auto f_zero   = Thyra::createMember(model->get_f_space());
    auto out_f = model->createOutArgs();

    Thyra::copy(x,x_scratch_.ptr());
    auto in_f = underlyingInArgs(x_scratch_,t,dt,0.0,1.0,nullptr);
    in_f.set_x_dot(x_dot_ones);
    out_f.set_f(row_sums);
    model->evalModel(in_f,out_f);

    Thyra::copy(x,x_scratch_.ptr());
    in_f.set_x_dot(x_dot_zero_);
    out_f.set_f(f_zero);
    model->evalModel(in_f,out_f);

    Thyra::Vp_StV(row_sums.ptr(),-1.0,*f_zero);

    if (inv_lumped_mass_.is_null()) {
      inv_lumped_mass_ = Thyra::createMember(model->get_f_space());
    }
    auto inv_data = getNonconstLocalData(*inv_lumped_mass_);
    auto rs_data  = getLocalData(*row_sums);
    auto d_data   = getLocalData(*diff_mask_);
    int num_bad = 0;
    for (LO i=0; i<static_cast<LO>(inv_data.size()); ++i) {
      if (d_data[i]!=0.0) {
        if (rs_data[i]>0.0) {
          inv_data[i] = 1.0/rs_data[i];
        } else {
          inv_data[i] = 0.0;
          ++num_bad;
        }
      } else {
        inv_data[i] = 0.0;
      }
    }
    int global_bad = 0;
    Teuchos::reduceAll(*getComm(inv_lumped_mass_->space()),Teuchos::REDUCE_SUM,1,&num_bad,&global_bad);
    TEUCHOS_TEST_FOR_EXCEPTION (global_bad>0, std::runtime_error,
        "Error! ExplicitODEModelEvaluator: " << global_bad << " differential dofs have a non-positive lumped mass.\n");
  } else {
    // M = df/dx_dot, i.e., W with alpha=1, beta=0.
    // Use a copy of x, since the model may modify it in place.
    if (mass_op_.is_null()) {
      mass_op_ = model->create_W_op();
    }
    Thyra::copy(x,x_scratch_.ptr());
    const auto in = underlyingInArgs(x_scratch_,t,dt,1.0,0.0,nullptr);
    auto out = model->createOutArgs();
    out.set_W_op(mass_op_);
    model->evalModel(in,out);

    // Identity on the rows/cols of the non-differential dofs (since x_dot=0 there, the
    // coupling of the differential rows with those columns is irrelevant), then setup
    // the solver.
    mass_eliminator_->apply(mass_op_);
    mass_lows_ = Thyra::linearOpWithSolve<ST>(*mass_lows_factory_,mass_op_.getConst());
  }
  mass_computed_ = true;
}

void ExplicitODEModelEvaluator::
applyInverseMass (const Thyra_Vector& r, Thyra_Vector& x_dot) const
{
  if (lumped_) {
    x_dot.assign(0.0);
    Thyra::ele_wise_prod<ST>(-1.0,*inv_lumped_mass_,r,Teuchos::ptrFromRef(x_dot));
  } else {
    // Separate lhs and rhs vectors (an iterative solver must not see them aliased)
    mass_sol_->assign(0.0);
    const auto status = mass_lows_->solve(Thyra::NOTRANS,r,mass_sol_.ptr());
    TEUCHOS_TEST_FOR_EXCEPTION (status.solveStatus==Thyra::SOLVE_STATUS_UNCONVERGED, std::runtime_error,
        "Error! ExplicitODEModelEvaluator: the mass matrix solve did not converge.\n");
    Thyra::V_StV<ST>(Teuchos::ptrFromRef(x_dot),-1.0,*mass_sol_);
    Thyra::ele_wise_scale<ST>(*diff_mask_,Teuchos::ptrFromRef(x_dot));
  }
}

// ------------------------------ Statistics ------------------------------ //

void ExplicitODEModelEvaluator::printStatistics (std::ostream& os) const
{
  os << "ExplicitODEModelEvaluator statistics:\n"
     << "  explicit rhs evaluations : " << num_rhs_evals_  << "\n"
     << "  algebraic solves         : " << num_solves_     << "\n"
     << "  newton iterations        : " << num_newton_its_ << "\n"
     << "  cache hits               : " << num_cache_hits_ << "\n";
}

} // namespace Albany
