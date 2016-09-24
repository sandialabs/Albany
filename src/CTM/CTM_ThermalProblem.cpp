#include "CTM_ThermalProblem.hpp"

namespace CTM {

ThermalProblem::ThermalProblem(
    const RCP<ParameterList>& params,
    RCP<ParamLib> const& param_lib,
    const int n_dims,
    RCP<const Teuchos::Comm<int> >& comm) :
  Albany::AbstractProblem(params, param_lib),
  num_dims(n_dims) {

    *out << "Problem name = Thermal Problem \n";
    this->setNumEquations(1);
    material_db = LCM::createMaterialDatabase(params, comm);
}

ThermalProblem::~ThermalProblem() {
}

void ThermalProblem::buildProblem(
    ArrayRCP<RCP<Albany::MeshSpecsStruct> > mesh_specs,
    Albany::StateManager& state_mgr) {
}

Teuchos::Array<RCP<const PHX::FieldTag> > ThermalProblem::buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm,
    const Albany::MeshSpecsStruct& mesh_specs,
    Albany::StateManager& state_mgr,
    Albany::FieldManagerChoice fm_choice,
    const RCP<ParameterList>& response_list) {

  // Call constructEvaluators<EvalT>() for specific evaluation types
  Albany::ConstructEvaluatorsOp<ThermalProblem> op(
      *this, fm, mesh_specs, state_mgr, fm_choice, response_list);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void ThermalProblem::constructDirichletEvaluators(
    const Albany::MeshSpecsStruct& mesh_specs) {
}

void ThermalProblem::constructNeumannEvaluators(
    const RCP<Albany::MeshSpecsStruct>& mesh_specs) {
}

void ThermalProblem::getAllocatedStates(
    ArrayRCP<ArrayRCP<RCP<FC> > > old_state,
    ArrayRCP<ArrayRCP<RCP<FC> > > new_state) const {
}

}
