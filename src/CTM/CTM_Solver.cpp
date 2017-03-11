#include "CTM_Solver.hpp"
#include "CTM_ThermalProblem.hpp"
#include "CTM_MechanicsProblem.hpp"
#include "CTM_SolutionInfo.hpp"
#include "CTM_LinearSolver.hpp"
#include "CTM_Assembler.hpp"

#include <Albany_DiscretizationFactory.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_APFDiscretization.hpp>

namespace CTM {

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->sublist("Time");
  p->sublist("Temperature Problem");
  p->sublist("Mechanics Problem");
  p->sublist("Discretization");
  p->sublist("Extra Discretization");
  p->sublist("Linear Algebra");
}

static void validate_params(RCP<const ParameterList> p) {
  assert(p->isSublist("Temperature Problem"));
  assert(p->isSublist("Mechanics Problem"));
  assert(p->isSublist("Discretization"));
  assert(p->isSublist("Extra Discretization"));
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
  initial_setup();
}

void Solver::set_params(RCP<ParameterList> p) {
  params = p;
  validate_params(p);
  t_params = rcpFromRef(params->sublist("Temperature Problem", true));
  m_params = rcpFromRef(params->sublist("Mechanics Problem", true));
  auto tp = params->sublist("Time");
  num_steps = tp.get<int>("Number of Steps");
  dt = tp.get<double>("Step Size");
  t_old = tp.get<double>("Initial Time");
  t_current = t_old + dt;
}

void Solver::initial_setup() {

  // initializations
  out = Teuchos::VerboseObjectBase::getDefaultOStream();
  param_lib = rcp(new ParamLib);
  dist_param_lib = rcp(new DistParamLib);
  t_state_mgr = rcp(new Albany::StateManager);
  m_state_mgr = rcp(new Albany::StateManager);
  t_sol_info = rcp(new SolutionInfo);
  m_sol_info = rcp(new SolutionInfo);

  // build the initial mesh specs
  auto disc_factory = rcp(new Albany::DiscretizationFactory(params, comm, false));
  mesh_specs = disc_factory->createMeshSpecs();
  int num_dims = mesh_specs[0]->numDim;

  // build the temperature problem
  t_problem = rcp(new ThermalProblem(t_params, param_lib, num_dims, comm));
  t_params->validateParameters(*(t_problem->getValidProblemParameters()), 0);
  t_problem->buildProblem(mesh_specs, *t_state_mgr);

  // build the mechanics problem
  m_problem = rcp(new CTM::MechanicsProblem(m_params, param_lib, num_dims, comm));
  m_params->validateParameters(*(m_problem->getValidProblemParameters()), 0);
  m_problem->buildProblem(mesh_specs, *m_state_mgr);

  // create the temperature discretization
  int neq = t_problem->numEquations();
  t_disc = disc_factory->createDiscretization(
      neq,
      t_problem->getSideSetEquations(),
      t_state_mgr->getStateInfoStruct(),
      t_state_mgr->getSideSetStateInfoStruct(),
      t_problem->getFieldRequirements(),
      t_problem->getSideSetFieldRequirements(),
      t_problem->getNullSpace());

  // create the mechanics discretization
  auto m_disc_params = rcpFromRef(params->sublist("Extra Discretization"));
  disc_factory->setDiscretizationParameters(m_disc_params);
  neq = m_problem->numEquations();
  m_disc = disc_factory->createDiscretization(
      neq,
      m_problem->getSideSetEquations(),
      m_state_mgr->getStateInfoStruct(),
      m_state_mgr->getSideSetStateInfoStruct(),
      m_problem->getFieldRequirements(),
      m_problem->getSideSetFieldRequirements(),
      m_problem->getNullSpace());

  // create the solution information
  t_sol_info->resize(t_disc, true);
  m_sol_info->resize(m_disc, false);
}

void Solver::solve() {
}

} // namespace CTM
