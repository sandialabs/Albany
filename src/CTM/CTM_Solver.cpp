#include "CTM_Solver.hpp"
#include "CTM_ThermalProblem.hpp"
#include <Albany_DiscretizationFactory.hpp>
#include <Albany_AbstractDiscretization.hpp>

namespace CTM {

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->sublist("Discretization");
  p->sublist("Temperature Problem");
//  p->sublist("Mechanics");
  p->sublist("Linear Algebra");
  p->sublist("Time");
}

static void validate_params(RCP<const ParameterList> p) {
  assert(p->isSublist("Discretization"));
  assert(p->isSublist("Temperature Problem"));
//  assert(p->isSublist("Mechanics"));
  assert(p->isSublist("Linear Algebra"));
  assert(p->isSublist("Time"));
}

Solver::Solver(
    RCP<const Teuchos_Comm> c,
    RCP<ParameterList> p) :
  comm(c),
  params(p) {

    validate_params(params);
    temp_params = rcpFromRef(params->sublist("Temperature Problem", true));

    initial_setup();
}

void Solver::initial_setup() {

  // create parameter libraries
  // note: we never intend to use these objects, we create them because they
  // are inputs to constructors for various other objects.
  param_lib = rcp(new ParamLib);
  dist_param_lib = rcp(new DistParamLib);

  // create the mesh specs struct
  bool explicit_scheme = false;
  disc_factory = rcp(new Albany::DiscretizationFactory(params, comm, false));
  mesh_specs = disc_factory->createMeshSpecs();

  // create the problem objects
  auto dim = mesh_specs[0]->numDim;
  t_problem = rcp(new ThermalProblem(temp_params, param_lib, dim, comm));

  temp_params->validateParameters(*(t_problem->getValidProblemParameters()),0);
  t_problem->buildProblem(mesh_specs, state_mgr);

  // create the initial discretization object
  auto neq = t_problem->numEquations();
  disc = disc_factory->createDiscretization(
      neq,
      t_problem->getSideSetEquations(),
      state_mgr.getStateInfoStruct(),
      state_mgr.getSideSetStateInfoStruct(),
      t_problem->getFieldRequirements(),
      t_problem->getSideSetFieldRequirements(),
      t_problem->getNullSpace());

}

void Solver::solve() {
}

}
