#include "CTM_Solver.hpp"
#include "CTM_ThermalProblem.hpp"
#include "CTM_MechanicsProblem.hpp"
#include "CTM_SolutionInfo.hpp"
#include "CTM_LinearSolver.hpp"
#include "CTM_Assembler.hpp"
#include "CTM_Adapter.hpp"

#include <Albany_DiscretizationFactory.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_APFDiscretization.hpp>

namespace CTM {

using Teuchos::rcp_dynamic_cast;

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->sublist("Time");
  p->sublist("Temperature Problem");
  p->sublist("Mechanics Problem");
  p->sublist("Discretization");
  p->sublist("Extra Discretization");
  p->sublist("Adaptation");
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
  if (params->isSublist("Adaptation"))
    adapt_params = rcpFromRef(params->sublist("Adaptation", true));
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
  *out << std::endl;

  // build the mechanics problem
  m_problem = rcp(new CTM::MechanicsProblem(m_params, param_lib, num_dims, comm));
  m_params->validateParameters(*(m_problem->getValidProblemParameters()), 0);
  m_problem->buildProblem(mesh_specs, *m_state_mgr);
  *out << std::endl;

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

  // build the assembler
  t_assembler = rcp(new Assembler(
        t_params, t_sol_info, t_problem, t_disc, t_state_mgr));
  m_assembler = rcp(new Assembler(
        m_params, m_sol_info, m_problem, m_disc, m_state_mgr));

  // set the state arrays
  *out << std::endl;
  t_state_mgr->setStateArrays(t_disc);
  m_state_mgr->setStateArrays(m_disc);

  // write the initial conditions for visualization
  auto apf_disc = rcp_dynamic_cast<Albany::APFDiscretization>(m_disc);
  apf_disc->writeAnySolutionToFile(0);

  // create the adapter if it is needed
  if (adapt_params != Teuchos::null)
    adapter = rcp(new Adapter(adapt_params, param_lib, t_state_mgr, m_state_mgr));

}

void Solver::solve_temp() {

  *out << "Solving thermal physics" << std::endl;

  // get linear solve parameters
  auto la_params = rcpFromRef(params->sublist("Linear Algebra"));

  // get the thermal solution info
  auto T = t_sol_info->owned->x;
  auto dTdt = t_sol_info->owned->x_dot;
  auto f = t_sol_info->owned->f;
  auto J = t_sol_info->owned->J;

  // create old vector + incremetal solution vector
  auto owned_map = t_disc->getMapT();
  auto T_old = rcp(new Tpetra_Vector(owned_map));
  auto delta_T = rcp(new Tpetra_Vector(owned_map));

  // compute fad coefficients
  double alpha = 1.0 / dt;
  double beta = 1.0;
  double omega = 0.0;

  // solve the linear system
  T_old->assign(*T);
  dTdt->update(alpha, *T, -alpha, *T_old, 0.0);
  t_assembler->assemble_system(alpha, beta, omega, t_current, t_old);
  f->scale(-1.0);
  delta_T->putScalar(0.0);
  solve_linear_system(la_params, J, delta_T, f);

  // perform updates
  T->update(1.0, *delta_T, 1.0);
  dTdt->update(alpha, *T, -alpha, *T_old, 0.0);
  t_assembler->assemble_state(t_current, t_old);
  t_state_mgr->updateStates();

  // save the solution to the mesh databse
  auto apf_disc = rcp_dynamic_cast<Albany::APFDiscretization>(t_disc);
  apf_disc->writeSolutionToMeshDatabaseT(*T, t_current, false);

}

void Solver::solve_mech() {

  *out << "Solving mechanics physics" << std::endl;

  // get linear solve parameters
  auto la_params = rcpFromRef(params->sublist("Linear Algebra"));

  // get the mechanics solution
  auto u = m_sol_info->owned->x;
  auto f = m_sol_info->owned->f;
  auto J = m_sol_info->owned->J;

  // compute the fad coefficients
  double alpha = 0.0;
  double beta = 1.0;
  double omega = 0.0;

  // solve the linear system
  u->putScalar(0.0);
  m_assembler->assemble_system(alpha, beta, omega, t_current, t_old);
  f->scale(-1.0);
  solve_linear_system(la_params, J, u, f);

  // perform updates
  m_assembler->assemble_state(t_current, t_old);
  m_state_mgr->updateStates();

  // save the solution to the mesh database
  auto apf_disc = rcp_dynamic_cast<Albany::APFDiscretization>(m_disc);
  apf_disc->writeSolutionToMeshDatabaseT(*u, t_current, false);

}

void Solver::adapt_mesh() {
  if (adapt_params == Teuchos::null) return;
  if (! adapter->should_adapt(t_current)) return;
  *out << "beginning mesh adaptation: " << std::endl;
  adapter->adapt(t_current);
  t_sol_info->resize(t_disc, true);
  m_sol_info->resize(m_disc, false);
  t_sol_info->owned->x = t_disc->getSolutionFieldT();
  m_sol_info->owned->x = m_disc->getSolutionFieldT();
  t_sol_info->scatter_x();
  m_sol_info->scatter_x();
}

void Solver::solve() {
  *out << std::endl;
  for (int step = 1; step <= num_steps; ++step) {

    *out << "*** Time Step: " << step << std::endl;
    *out << "*** from time: " << t_old << std::endl;
    *out << "*** to time: " << t_current << std::endl;

    // perform the analysis
    solve_temp();
    solve_mech();

    // save the solution to file after analysis
    auto apf_disc = rcp_dynamic_cast<Albany::APFDiscretization>(m_disc);
    apf_disc->writeAnySolutionToFile(t_current);

    // adapt the mesh if needed
    adapt_mesh();

    // update the time information
    t_old = t_current;
    t_current += dt;
  }
}

} // namespace CTM
