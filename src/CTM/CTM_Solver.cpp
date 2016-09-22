#include "CTM_Solver.hpp"
#include <Albany_DiscretizationFactory.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_HeatProblem.hpp>

namespace CTM {

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->sublist("Discretization");
  p->sublist("Temperature");
//  p->sublist("Mechanics");
  p->sublist("Linear Algebra");
  p->sublist("Time");
}

static void validate_params(RCP<const ParameterList> p) {
  assert(p->isSublist("Discretization"));
  assert(p->isSublist("Temperature"));
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
    temp_params = rcpFromRef(params->sublist("Temperature", true));

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
  auto d = mesh_specs[0]->numDim;
  temp_problem = rcp(new Albany::HeatProblem(temp_params, param_lib, d, comm));
  temp_params->validateParameters(*(temp_problem->getValidProblemParameters()),0);

}

void Solver::solve() {
}

}
