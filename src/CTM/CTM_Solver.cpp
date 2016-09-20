#include "CTM_Solver.hpp"

namespace CTM {

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->sublist("Discretization");
  p->sublist("Temperature");
  p->sublist("Mechanics");
  p->sublist("Linear Algebra");
  p->sublist("Time");
}

static void validate_params(RCP<const ParameterList> p) {
  assert(p->isSublist("Discretization"));
  assert(p->isSublist("Temperature"));
  assert(p->isSublist("Mechanics"));
  assert(p->isSublist("Linear Algebra"));
  assert(p->isSublist("Time"));
}

Solver::Solver(
    RCP<const Teuchos_Comm> c,
    RCP<ParameterList> p) :
  comm(c),
  params(p) {
    validate_params(params);
}

void Solver::solve() {
}

}
