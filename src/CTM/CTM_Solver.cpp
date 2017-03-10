#include "CTM_Solver.hpp"
#include "CTM_ThermalProblem.hpp"
#include "CTM_MechanicsProblem.hpp"
#include "CTM_SolutionInfo.hpp"
#include "CTM_LinearSolver.hpp"

#include <Albany_DiscretizationFactory.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_APFDiscretization.hpp>

namespace CTM {

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->sublist("Temperature Problem");
  p->sublist("Thermal Discretization");
  p->sublist("Mechanics Problem");
  p->sublist("Mechanics Discretization");
  p->sublist("Linear Algebra");
  p->sublist("Time");
}

static void validate_params(RCP<const ParameterList> p) {
  assert(p->isSublist("Temperature Problem"));
  assert(p->isSublist("Mechanics Problem"));
  assert(p->isSublist("Thermal Discretization"));
  assert(p->isSublist("Mechanics Discretization"));
  assert(p->isSublist("Linear Algebra"));
  assert(p->isSublist("Time"));

  auto time_params = p->sublist("Time");
  assert(time_params.isType<double>("Initial Time"));
  assert(time_params.isType<double>("Step Size"));
  assert(time_params.isType<int>("Number of Steps"));

  auto la_params = p->sublist("Linear Algebra");
  assert(la_params.isType<double>("Linear Tolerance"));
  assert(la_params.isType<int>("Linear Max Iterations"));
  assert(la_params.isType<int>("Linear Krylov Size"));
}

Solver::Solver(RCP<const Teuchos_Comm> c, RCP<ParameterList> p) {
  comm = c;
  set_params(p);
}

void Solver::initial_setup() {
}

void Solver::set_params(RCP<ParameterList> p) {
  params = p;
  validate_params(p);
  temp_params = rcpFromRef(params->sublist("Temperature Problem", true));
  mech_params = rcpFromRef(params->sublist("Mechanics Problem", true));
  auto tp = params->sublist("Time");
  num_steps = tp.get<int>("Number of Steps");
  dt = tp.get<double>("Step Size");
  t_old = tp.get<double>("Initial Time");
  t_current = t_old + dt;
}

void Solver::solve() {
}

} // namespace CTM
