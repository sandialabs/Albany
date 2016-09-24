#include "CTM_ThermalProblem.hpp"

namespace CTM {

ThermalProblem::ThermalProblem(
    const RCP<ParameterList>& params,
    RCP<ParamLib> const& param_lib,
    const int num_dims,
    RCP<const Teuchos::Comm<int> >& comm) :
  Albany::AbstractProblem(params, param_lib) {
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
