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
ConsistentInitialStateSolver (const MaskFunction&                     mask_function,
                              const Teuchos::ParameterList&           nox_params,
                              const Teuchos::RCP<Thyra_LOWS_Factory>& lows_factory)
 : mask_function_(mask_function)
 , nox_params_(nox_params)
 , lows_factory_(lows_factory)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!mask_function_, std::invalid_argument,
      "Error! ConsistentInitialStateSolver: the function of the mask of the algebraic dofs is empty.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (lows_factory_.is_null(), std::invalid_argument,
      "Error! ConsistentInitialStateSolver: the linear solver factory is null.\n");
}

ConsistentInitialStateSolver::~ConsistentInitialStateSolver () = default;

void ConsistentInitialStateSolver::
buildSolver (const Teuchos::RCP<Thyra_ModelEvaluator>& model)
{
  model_   = model;
  x_space_ = model->get_x_space();

  solve_mask_ = mask_function_();
  TEUCHOS_TEST_FOR_EXCEPTION (solve_mask_.is_null(), std::logic_error,
      "Error! ConsistentInitialStateSolver: the mask of the algebraic dofs is null.\n");

  // The constraint model checks that the mask is in the solution space of the model
  constraint_ = Teuchos::rcp(new AlgebraicConstraintModelEvaluator(model,solve_mask_));
  auto constraint_with_solve = Teuchos::rcp(
      new Thyra::DefaultModelEvaluatorWithSolveFactory<ST>(constraint_,lows_factory_));

  // NOX modifies its parameter list (status tests, row sum scaling objects, ...),
  // so give each solver its own copy
  nox_ = Teuchos::rcp(new Thyra::NOXNonlinearSolver());
  nox_->setParameterList(Teuchos::rcp(new Teuchos::ParameterList(nox_params_)));
  nox_->setModel(constraint_with_solve);
}

void ConsistentInitialStateSolver::
solve (const Teuchos::RCP<Thyra_ModelEvaluator>& model,
       const Thyra_InArgs&                       inArgs,
       Thyra_Vector&                             x0)
{
  using MEB = Thyra::ModelEvaluatorBase;

  TEUCHOS_TEST_FOR_EXCEPTION (model.is_null(), std::invalid_argument,
      "Error! ConsistentInitialStateSolver::solve: the model is null.\n");

  // The mask, the constraint model and the Newton solver are tied to the model and to its
  // solution space: build them at the first solve, and again if either changed (e.g., the
  // solution space of the Albany model changes with mesh adaptation)
  if (model.get()!=model_.get() || model->get_x_space().get()!=x_space_.get()) {
    buildSolver(model);
  }

  // The nonlinear solver gives the constraint model no parameters: they come from here
  Teuchos::Array<Teuchos::RCP<const Thyra_Vector>> p(inArgs.Np());
  for (int l=0; l<inArgs.Np(); ++l) {
    p[l] = inArgs.get_p(l);
  }
  const ST t  = inArgs.supports(MEB::IN_ARG_t)         ? inArgs.get_t()         : 0.0;
  const ST dt = inArgs.supports(MEB::IN_ARG_step_size) ? inArgs.get_step_size() : 0.0;
  constraint_->setParameters(p);
  constraint_->setTime(t);
  constraint_->setStepSize(dt);

  // Only the non-algebraic dofs of x0 are used as fixed values; the algebraic dofs
  // of x0 are the initial guess
  constraint_->setFixedValues(x0);

  const auto status = nox_->solve(&x0);
  TEUCHOS_TEST_FOR_EXCEPTION (status.solveStatus!=Thyra::SOLVE_STATUS_CONVERGED, std::runtime_error,
      "Error! ConsistentInitialStateSolver: the solve for the algebraic dofs of the initial state\n"
      "       did not converge (t = " << t << ").\n"
      "       Revisit the nonlinear solver settings (Problem->Consistent Initial State->NOX),\n"
      "       or the initial guess of the algebraic dofs (the initial condition).\n");
  ++num_solves_;
}

} // namespace Albany
