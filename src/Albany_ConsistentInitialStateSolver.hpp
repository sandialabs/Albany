//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CONSISTENT_INITIAL_STATE_SOLVER_HPP
#define ALBANY_CONSISTENT_INITIAL_STATE_SOLVER_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_AlgebraicConstraintModelEvaluator.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <map>

namespace Thyra { class NOXNonlinearSolver; }

namespace Albany {

/*
 * Makes the initial state of a semi-explicit index-1 DAE consistent: solves the
 * algebraic equations
 *
 *   f(x_dot=0, [x_a, x_r], t0)_a = 0
 *
 * for the algebraic dofs x_a of the initial state, given its other dofs x_r and the
 * parameters of the time integration (e.g., the FO velocity for the initial ice
 * thickness of LandIce::StokesFOThickness). This is a Newton solve
 * (Thyra::NOXNonlinearSolver) on AlgebraicConstraintModelEvaluator, like the
 * algebraic solves of ExplicitODEModelEvaluator.
 *
 * Albany::SolverFactory gives it to Piro::TempusSolver (setConsistentInitialStateFunction),
 * which calls solve before each forward time integration, for the current parameters (see
 * Problem->"Consistent Initial State"). Implicit steppers do not use the algebraic dofs
 * of the initial state (except as an initial guess), but the adjoint sensitivities of
 * Tempus (continuous adjoint) do: the last step of the adjoint integration evaluates the
 * model at the initial state, so inconsistent algebraic dofs give a gradient error that
 * is O(dt), but with a large constant.
 *
 * The Piro solvers may call solve with different models evaluating the same residual
 * (the model of the Piro::TempusSolver, and the product model of the transient
 * optimization). Each model gets its own constraint model and Newton solver, built at
 * its first solve.
 */
class ConsistentInitialStateSolver
{
public:
  // solve_mask:   vector in the x space; entries != 0 flag the algebraic dofs, the
  //               dofs to solve for. The other dofs keep their values.
  // nox_params:   parameters of the Newton solves (a copy is made for each solver)
  // lows_factory: linear solver of the Newton solves
  ConsistentInitialStateSolver (const Teuchos::RCP<const Thyra_Vector>& solve_mask,
                                const Teuchos::ParameterList&           nox_params,
                                const Teuchos::RCP<Thyra_LOWS_Factory>& lows_factory);

  ~ConsistentInitialStateSolver ();

  //! Solve for the algebraic dofs of x0, in place, given its other dofs and the time,
  //! step size and parameters in inArgs (InArgs of model). Throws if the solve fails.
  void solve (const Teuchos::RCP<Thyra_ModelEvaluator>& model,
              const Thyra_InArgs&                       inArgs,
              Thyra_Vector&                             x0);

  Teuchos::RCP<const Thyra_Vector> getMask () const { return solve_mask_; }

  //! Number of (successful) solves so far
  int numSolves () const { return num_solves_; }

private:
  struct Solver {
    Teuchos::RCP<AlgebraicConstraintModelEvaluator> constraint;
    Teuchos::RCP<Thyra::NOXNonlinearSolver>         nox;
  };

  Teuchos::RCP<const Thyra_Vector>  solve_mask_;
  Teuchos::ParameterList            nox_params_;
  Teuchos::RCP<Thyra_LOWS_Factory>  lows_factory_;

  // Keyed by the model. The solver holds an RCP to the model, so the key stays valid.
  std::map<const Thyra_ModelEvaluator*, Solver> solvers_;

  int num_solves_ = 0;
};

} // namespace Albany

#endif // ALBANY_CONSISTENT_INITIAL_STATE_SOLVER_HPP
