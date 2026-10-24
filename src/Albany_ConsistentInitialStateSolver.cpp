//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ConsistentInitialStateSolver.hpp"

#include "Thyra_NonlinearSolver_NOX.hpp"
#include "Thyra_DefaultModelEvaluatorWithSolveFactory.hpp"
#include "Teuchos_TestForException.hpp"

namespace Albany {

ConsistentInitialStateSolver::
ConsistentInitialStateSolver (const Teuchos::RCP<const Thyra_Vector>& solve_mask,
                              const Teuchos::ParameterList&           nox_params,
                              const Teuchos::RCP<Thyra_LOWS_Factory>& lows_factory)
 : solve_mask_(solve_mask)
 , nox_params_(nox_params)
 , lows_factory_(lows_factory)
{
  TEUCHOS_TEST_FOR_EXCEPTION (solve_mask_.is_null(), std::invalid_argument,
      "Error! ConsistentInitialStateSolver: the mask of the algebraic dofs is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (lows_factory_.is_null(), std::invalid_argument,
      "Error! ConsistentInitialStateSolver: the linear solver factory is null.\n");
}

ConsistentInitialStateSolver::~ConsistentInitialStateSolver () = default;

void ConsistentInitialStateSolver::
solve (const Teuchos::RCP<Thyra_ModelEvaluator>& model,
       const Thyra_InArgs&                       inArgs,
       Thyra_Vector&                             x0)
{
  using MEB = Thyra::ModelEvaluatorBase;

  TEUCHOS_TEST_FOR_EXCEPTION (model.is_null(), std::invalid_argument,
      "Error! ConsistentInitialStateSolver::solve: the model is null.\n");

  Solver& s = solvers_[model.get()];
  if (s.constraint.is_null()) {
    // The constraint model checks that the mask is in the solution space of the model
    s.constraint = Teuchos::rcp(new AlgebraicConstraintModelEvaluator(model,solve_mask_));
    auto constraint_with_solve = Teuchos::rcp(
        new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(s.constraint,lows_factory_));

    // NOX modifies its parameter list (status tests, row sum scaling objects, ...),
    // so give each solver its own copy
    s.nox = Teuchos::rcp(new Thyra::NOXNonlinearSolver());
    s.nox->setParameterList(Teuchos::rcp(new Teuchos::ParameterList(nox_params_)));
    s.nox->setModel(constraint_with_solve);
  }

  // The nonlinear solver gives the constraint model no parameters: they come from here
  Teuchos::Array<Teuchos::RCP<const Thyra_Vector>> p(inArgs.Np());
  for (int l=0; l<inArgs.Np(); ++l) {
    p[l] = inArgs.get_p(l);
  }
  const ST t  = inArgs.supports(MEB::IN_ARG_t)         ? inArgs.get_t()         : 0.0;
  const ST dt = inArgs.supports(MEB::IN_ARG_step_size) ? inArgs.get_step_size() : 0.0;
  s.constraint->setParameters(p);
  s.constraint->setTime(t);
  s.constraint->setStepSize(dt);

  // Only the non-algebraic dofs of x0 are used as fixed values; the algebraic dofs
  // of x0 are the initial guess
  s.constraint->setFixedValues(x0);

  const auto status = s.nox->solve(&x0);
  TEUCHOS_TEST_FOR_EXCEPTION (status.solveStatus!=Thyra::SOLVE_STATUS_CONVERGED, std::runtime_error,
      "Error! ConsistentInitialStateSolver: the solve for the algebraic dofs of the initial state\n"
      "       did not converge (t = " << t << ").\n"
      "       Revisit the nonlinear solver settings (Problem->Consistent Initial State->NOX),\n"
      "       or the initial guess of the algebraic dofs (the initial condition).\n");
  ++num_solves_;
}

} // namespace Albany
