//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "SolutionManager.hpp"

#include "InitialCondition.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_ModelEvaluator.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"

#include "Thyra_ModelEvaluatorDelegatorBase.hpp"

namespace Albany {

SolutionManager::SolutionManager(
    Teuchos::RCP<Teuchos::ParameterList> const& appParams,
    Teuchos::RCP<Thyra_Vector const> const&     initial_guess,
    Teuchos::RCP<ParamLib> const&               param_lib,
    Teuchos::RCP<Albany::AbstractDiscretization>const & disc,
    Teuchos::RCP<Teuchos_Comm const> const&     comm)
    : num_time_deriv(appParams->sublist("Discretization")
                         .get<int>("Number Of Time Derivatives")),
      appParams_(appParams),
      disc_(disc),
      paramLib_(param_lib),
      comm_(comm),
      out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // Create problem PL
  Teuchos::RCP<Teuchos::ParameterList> problemParams =
      Teuchos::sublist(appParams_, "Problem", true);

  num_params_ = Albany::CalculateNumberParams(problemParams);

  // Want the initial time in the parameter library to be correct
  // if this is a restart solution
  // MJJ (12/06/16) I'll go an remove this conditional "if
  // (disc_->hasRestartSolution())" Independent of restarting analysis you want
  // to have the capability of specify an initial time. Not doing so was causing
  // that the initial time was always set to zero and the latter cause many
  // problem because for time 0, dt is set up as 1.0e-15, creating serious
  // problem when a body source is specified at time 0. For example, in a heat
  // transfer problem at time 0 and specifying a heat source the problem may
  // corresponds to a "thermal shock" problem, causing some instabilities in the
  // time integrator.
  if (paramLib_->isParameter("Time")) {
    double initialValue = 0.0;

    if (appParams->sublist("Problem").get<std::string>(
            "Solution Method", "Steady") == "Continuation") {
      initialValue = appParams->sublist("Piro")
                         .sublist("LOCA")
                         .sublist("Stepper")
                         .get<double>("Initial Value", 0.0);
    } else if (
        appParams->sublist("Problem").get<std::string>(
            "Solution Method", "Steady") == "Transient") {
      initialValue = appParams->sublist("Piro")
                         .sublist("Trapezoid Rule")
                         .get<double>("Initial Time", 0.0);
    }
    paramLib_->setRealValue<PHAL::AlbanyTraits::Residual>("Time", initialValue);
  }

  {
    auto owned_vs      = disc_->getVectorSpace();
    auto overlapped_vs = disc_->getOverlapVectorSpace();

    overlapped_soln = Thyra::createMembers(overlapped_vs, num_time_deriv + 1);
    if (num_params_ > 0) {
      overlapped_soln_dxdp = Thyra::createMembers(overlapped_vs, num_params_);
    } else {
      overlapped_soln_dxdp = Teuchos::null;
    }

    // TODO: ditch the overlapped_*T and keep only overlapped_*.
    //       You need to figure out how to pass the graph in a Tpetra-free way
    //       though...
    overlapped_f = Thyra::createMember(overlapped_vs);

    // This call allocates the non-overlapped MV
    current_soln = disc_->getSolutionMV();

    // Create the CombineAndScatterManager for handling distributed memory linear
    // algebra communications
    cas_manager = Albany::createCombineAndScatterManager(owned_vs, overlapped_vs);
  }

  auto pbParams = Teuchos::sublist(appParams_, "Problem", true);

  if (Teuchos::nonnull(initial_guess)) {
    current_soln->col(0)->assign(*initial_guess);
  } else {
    cas_manager->scatter(
        current_soln->col(0),
        overlapped_soln->col(0),
        Albany::CombineMode::INSERT);
    InitialConditions(
        overlapped_soln->col(0),
        disc,
        pbParams->sublist("Initial Condition"));
    cas_manager->combine(
        overlapped_soln->col(0),
        current_soln->col(0),
        Albany::CombineMode::INSERT);

    if (num_time_deriv > 0) {
      cas_manager->scatter(
          current_soln->col(1),
          overlapped_soln->col(1),
          Albany::CombineMode::INSERT);
      InitialConditions(
          overlapped_soln->col(1),
          disc,
          pbParams->sublist("Initial Condition Dot"));
      cas_manager->combine(
          overlapped_soln->col(1),
          current_soln->col(1),
          Albany::CombineMode::INSERT);
    }

    if (num_time_deriv > 1) {
      cas_manager->scatter(
          current_soln->col(2),
          overlapped_soln->col(2),
          Albany::CombineMode::INSERT);
      InitialConditions(
          overlapped_soln->col(2),
          disc,
          pbParams->sublist("Initial Condition DotDot"));
      cas_manager->combine(
          overlapped_soln->col(1),
          current_soln->col(1),
          Albany::CombineMode::INSERT);
    }
  }
}

Teuchos::RCP<const Thyra_Vector>
SolutionManager::updateAndReturnOverlapSolution(
    const Thyra_Vector& solution /* not overlapped */)
{
  cas_manager->scatter(
      solution, *overlapped_soln->col(0), Albany::CombineMode::INSERT);
  return overlapped_soln->col(0);
}

Teuchos::RCP<const Thyra_Vector>
SolutionManager::updateAndReturnOverlapSolutionDot(
    const Thyra_Vector& solution_dot /* not overlapped */)
{
  cas_manager->scatter(
      solution_dot, *overlapped_soln->col(1), Albany::CombineMode::INSERT);
  return overlapped_soln->col(1);
}

Teuchos::RCP<const Thyra_Vector>
SolutionManager::updateAndReturnOverlapSolutionDotDot(
    const Thyra_Vector& solution_dotdot /* not overlapped */)
{
  cas_manager->scatter(
      solution_dotdot, *overlapped_soln->col(2), Albany::CombineMode::INSERT);
  return overlapped_soln->col(2);
}

Teuchos::RCP<const Thyra_MultiVector>
SolutionManager::updateAndReturnOverlapSolutionMV(
    const Thyra_MultiVector& solution /* not overlapped */)
{
  cas_manager->scatter(solution, *overlapped_soln, Albany::CombineMode::INSERT);
  return overlapped_soln;
}

Teuchos::RCP<Thyra_MultiVector>
SolutionManager::updateAndReturnOverlapSolutionDxDp(
    const Thyra_MultiVector& solution_dxdp /* not overlapped */)
{
  cas_manager->scatter(solution_dxdp, *overlapped_soln_dxdp, Albany::CombineMode::INSERT);
  return overlapped_soln_dxdp;
}

void
SolutionManager::scatterX(
    const Thyra_MultiVector& solution /* not overlapped */,
    const Teuchos::Ptr<const Thyra_MultiVector> solution_dxdp /* not overlapped */)
{
  cas_manager->scatter(solution, *overlapped_soln, Albany::CombineMode::INSERT);
  if (solution_dxdp != Teuchos::null) {
    const int np = solution_dxdp->domain()->dim();
    if (np != num_params_) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "SolutionManager::scatterX error: size dxdp (" <<
          np << ") != num_params (" << num_params_ << ").\n");
    }
    cas_manager->scatter(*solution_dxdp, *overlapped_soln_dxdp, Albany::CombineMode::INSERT);
  }
}

void
SolutionManager::scatterX(
    const Thyra_Vector&                    x,
    const Teuchos::Ptr<const Thyra_Vector> x_dot,
    const Teuchos::Ptr<const Thyra_Vector> x_dotdot,
    const Teuchos::Ptr<const Thyra_MultiVector> dxdp)
{
  cas_manager->scatter(
      x, *overlapped_soln->col(0), Albany::CombineMode::INSERT);

  if (!x_dot.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        overlapped_soln->domain()->dim() < 2,
        std::logic_error,
        "SolutionManager error: x_dot defined but only a single "
        "solution vector is available");
    cas_manager->scatter(
        *x_dot, *overlapped_soln->col(1), Albany::CombineMode::INSERT);
  }

  if (!x_dotdot.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        overlapped_soln->domain()->dim() < 3,
        std::logic_error,
        "SolutionManager error: x_dotdot defined but only two solution "
        "vectors are available");
    cas_manager->scatter(
        *x_dotdot, *overlapped_soln->col(2), Albany::CombineMode::INSERT);
  }
  if (!dxdp.is_null()) {
    const int np = dxdp->domain()->dim();
    if (np != num_params_) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "SolutionManager::scatterX error: size dxdp (" <<
          np << ") != num_params (" << num_params_ << ").\n");
    }
    cas_manager->scatter(
        *dxdp, *overlapped_soln_dxdp, Albany::CombineMode::INSERT);
  }
}

}  // namespace Albany
