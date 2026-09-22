//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EXPLICIT_ODE_MODEL_EVALUATOR_HPP
#define ALBANY_EXPLICIT_ODE_MODEL_EVALUATOR_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_AlgebraicConstraintModelEvaluator.hpp"

#include "Thyra_ModelEvaluatorDelegatorBase.hpp"
#include "Teuchos_ParameterList.hpp"

#include <ostream>

namespace Thyra { class NOXNonlinearSolver; }

namespace Albany {

/*
 * Whether the Tempus stepper configured in the given Piro->Tempus parameter list is
 * an explicit stepper for first-order ODEs, i.e., whether Tempus will evaluate the
 * model as x_dot = f(x,t) (see Tempus::StepperExplicit). Steppers for second-order
 * ODEs (Newmark Explicit a-Form, Leapfrog), IMEX steppers (partly implicit) and
 * steppers with sub-steppers (Operator Split, Subcycling, whose sub-stepper may be
 * implicit) do not qualify. The parameter list is not modified, and an unrecognized
 * stepper type returns false (Piro/Tempus will report it).
 */
bool usesExplicitTempusStepper (const Teuchos::ParameterList& piroTempusParams);

/*
 * Turns a semi-explicit index-1 DAE, written in Albany's implicit form
 *
 *     f(x_dot, x, t) = 0,   x = [x_a, x_d, x_c],
 *
 *   x_a: algebraic dofs    (no x_dot dependence, e.g. the FO velocity)
 *   x_d: differential dofs (f_d = M x_dot_d + r_d(x,t), e.g. the ice thickness)
 *   x_c: constant dofs     (neither: e.g. off-level thickness dofs, Dirichlet dofs)
 *
 * into the explicit ODE that Tempus' explicit steppers expect:
 *
 *     x_dot = F(x,t),   F_d = -M^{-1} r_d(x*,t),   F_a = F_c = 0.
 *
 * The algebraic constraint f_a(0,x,t) = 0 (e.g. a PDE: the FO velocity equations) is
 * *eliminated* from the time integration: at every evaluation of the right-hand side,
 * it is solved for x_a, given the current x_d and x_c, so that x* = [x_a(x_d,x_c,t),
 * x_d, x_c]. The solve is a Newton solve on AlgebraicConstraintModelEvaluator (with
 * Thyra::NOXNonlinearSolver). This is the "state space form" of the DAE, and applying
 * an s-stage explicit RK method to it is a half-explicit RK method: s nonlinear solves
 * per step, with the RK order preserved for x_d, as long as the algebraic solves are
 * accurate compared to the time discretization error.
 *
 * Since the model is already an explicit ODE (as far as Tempus is concerned, its mass
 * matrix is the identity), Piro must *not* wrap it in an InvertMassMatrixDecorator:
 * Albany::SolverFactory sets "Invert Mass Matrix" to false in the Tempus list.
 *
 * The model keeps the full x space, so output, restart, SDBCs and responses work
 * as for the monolithic model. The algebraic entries of the x that Tempus stores
 * are *not* evolved (F_a = 0): call syncAlgebraicState at the end of each step
 * (e.g., from Albany::PiroTempusObserver) to make them consistent with x_d. The
 * result is cached, so the first stage of the next step does not solve again.
 *
 * Parameters (see getValidParameters):
 *   "Algebraic Update": "Every Stage" (default) or "Once Per Step". The latter
 *        solves x_a only in syncAlgebraicState (and at the first evaluation), and
 *        reuses it for all stages: one solve per step, first order in the coupling.
 *   "Lump Mass Matrix": lumped mass for the differential dofs (row sums, computed as
 *        f(x_dot=1)-f(x_dot=0)) instead of the consistent mass (W with alpha=1,
 *        beta=0, rows/cols of the non-differential dofs set to identity, solved with
 *        the mass LOWS factory). Default false, as in Piro::TempusSolver, which reads
 *        the same option for its own mass matrix inversion.
 *   "Constant Mass Matrix": compute M only once (default false, as in Piro).
 *
 * Requirements on the underlying model: InArgs x, x_dot, t, alpha, beta (and
 * optionally step_size); OutArgs f, W_op; f affine in x_dot (M may depend on x, t).
 */
class ExplicitODEModelEvaluator : public Thyra::ModelEvaluatorDelegatorBase<ST>
{
public:
  ExplicitODEModelEvaluator (const Teuchos::RCP<Thyra_ModelEvaluator>&   model,
                             const Teuchos::RCP<const Thyra_Vector>&     algebraic_mask,
                             const Teuchos::RCP<const Thyra_Vector>&     differential_mask,
                             const Teuchos::RCP<Teuchos::ParameterList>& params,
                             const Teuchos::RCP<Teuchos::ParameterList>& noxParams,
                             const Teuchos::RCP<Thyra_LOWS_Factory>&     algebraicLowsFactory,
                             const Teuchos::RCP<Thyra_LOWS_Factory>&     massLowsFactory = Teuchos::null);

  ~ExplicitODEModelEvaluator ();

  static Teuchos::RCP<const Teuchos::ParameterList> getValidParameters ();

  //! Rebuild everything that depends on the solution space: masks, work vectors,
  //! cache, mass matrix and algebraic solver. To be called after mesh adaptation,
  //! with the masks of the *adapted* discretization (see Albany::PiroTempusObserver).
  void reinitialize (const Teuchos::RCP<const Thyra_Vector>& algebraic_mask,
                     const Teuchos::RCP<const Thyra_Vector>& differential_mask);

  //! Solve for the algebraic dofs of x, given its other dofs, at time t (in place).
  //! Reuses the cached solve if the non-algebraic part of x and t match it.
  //! Values prescribed in place by the model (e.g., SDBCs) are also written in x.
  void syncAlgebraicState (Thyra_Vector& x, const ST t) const;

  //! Last state with consistent algebraic dofs (null before the first solve)
  Teuchos::RCP<const Thyra_Vector> getLastConsistentState () const;
  ST getLastConsistentStateTime () const { return cache_t_; }

  Teuchos::RCP<const Thyra_Vector> getAlgebraicMask    () const { return alg_mask_;  }
  Teuchos::RCP<const Thyra_Vector> getDifferentialMask () const { return diff_mask_; }

  /** \name Statistics */
  //@{
  int numAlgebraicSolves  () const { return num_solves_; }
  int numNewtonIterations () const { return num_newton_its_; }
  int numCacheHits        () const { return num_cache_hits_; }
  int numRhsEvaluations   () const { return num_rhs_evals_; }
  void printStatistics (std::ostream& os) const;
  //@}

  /** \name Overridden from Thyra::ModelEvaluator */
  //@{
  Thyra_InArgs createInArgs () const override;
  Thyra_InArgs getNominalValues () const override;
  Thyra_InArgs getLowerBounds () const override;
  Thyra_InArgs getUpperBounds () const override;

  // This is an explicit ODE model: no W
  Teuchos::RCP<Thyra_LinearOp>           create_W_op   () const override { return Teuchos::null; }
  Teuchos::RCP<Thyra_LOWS>               create_W      () const override;
  Teuchos::RCP<const Thyra_LOWS_Factory> get_W_factory () const override { return Teuchos::null; }
  Teuchos::RCP<Thyra_Preconditioner>     create_W_prec () const override { return Teuchos::null; }
  //@}

private:

  /** \name Overridden from Thyra::ModelEvaluatorDefaultBase */
  //@{
  Thyra_OutArgs createOutArgsImpl () const override;

  void evalModelImpl (const Thyra_InArgs&  inArgs,
                      const Thyra_OutArgs& outArgs) const override;
  //@}

  // Setup (also used by reinitialize)
  void setMasks (const Teuchos::RCP<const Thyra_Vector>& algebraic_mask,
                 const Teuchos::RCP<const Thyra_Vector>& differential_mask);
  void allocateWorkVectors ();
  void createAlgebraicSolver ();

  // Solve for the algebraic dofs of x (in place), unless the cache can be used
  void makeConsistent (Thyra_Vector& x, const ST t, const ST dt) const;

  bool matchesCache (const Thyra_Vector& x, const ST t) const;
  void copyFromCache (Thyra_Vector& x, const bool also_prescribed) const;

  // Build InArgs for the underlying model (non-null zero x_dot, given alpha/beta)
  Thyra_InArgs underlyingInArgs (const Teuchos::RCP<const Thyra_Vector>& x,
                                 const ST t, const ST dt,
                                 const ST alpha, const ST beta,
                                 const Thyra_InArgs* inArgs) const;

  void computeMassMatrix (const Thyra_Vector& x, const ST t, const ST dt) const;
  void applyInverseMass  (const Thyra_Vector& r, Thyra_Vector& x_dot) const;

  // Masks
  Teuchos::RCP<const Thyra_Vector>  alg_mask_;
  Teuchos::RCP<const Thyra_Vector>  diff_mask_;
  Teuchos::RCP<Thyra_Vector>        non_diff_mask_;   // 1 - diff_mask

  // Options
  bool lagged_        = false;
  bool lumped_        = false;
  bool constant_mass_ = false;

  // Algebraic solve
  Teuchos::RCP<AlgebraicConstraintModelEvaluator>  constraint_model_;
  Teuchos::RCP<Thyra::NOXNonlinearSolver>          nox_solver_;
  Teuchos::RCP<Thyra_LOWS_Factory>                 alg_lows_factory_;
  Teuchos::RCP<Teuchos::ParameterList>             nox_params_;

  // Mass matrix
  Teuchos::RCP<Thyra_LOWS_Factory>          mass_lows_factory_;
  mutable Teuchos::RCP<Thyra_LinearOp>      mass_op_;
  mutable Teuchos::RCP<Thyra_LOWS>          mass_lows_;
  mutable Teuchos::RCP<Thyra_Vector>        inv_lumped_mass_;
  mutable bool                              mass_computed_ = false;
  Teuchos::RCP<MaskedDofsEliminator>        mass_eliminator_;

  // Cache of the last algebraic solve:
  //  - cache_key_: the state as given (before the solve)
  //  - cache_x_  : the consistent state (after the solve, incl. in-place BC values)
  mutable Teuchos::RCP<Thyra_Vector>  cache_key_;
  mutable Teuchos::RCP<Thyra_Vector>  cache_x_;
  mutable ST                          cache_t_     = 0.0;
  mutable bool                        cache_valid_ = false;
  mutable ST                          last_dt_     = 0.0;

  // Work vectors
  Teuchos::RCP<Thyra_Vector>  x_work_;
  Teuchos::RCP<Thyra_Vector>  x_sync_;
  Teuchos::RCP<Thyra_Vector>  x_scratch_;
  Teuchos::RCP<Thyra_Vector>  x_dot_zero_;
  Teuchos::RCP<Thyra_Vector>  residual_;
  Teuchos::RCP<Thyra_Vector>  mass_sol_;

  // Statistics
  mutable int num_solves_        = 0;
  mutable int num_newton_its_    = 0;
  mutable int num_cache_hits_    = 0;
  mutable int num_rhs_evals_     = 0;
};

} // namespace Albany

#endif // ALBANY_EXPLICIT_ODE_MODEL_EVALUATOR_HPP
