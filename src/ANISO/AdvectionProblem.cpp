//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AdvectionProblem.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"

Albany::AdvectiveProblem::AdvectiveProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<ParamLib>& param_lib,
    const int num_dims,
 	  Teuchos::RCP<const Teuchos::Comm<int> >& commT) :
  Albany::AbstractProblem(params_, param_lib),
  num_dims_(num_dims) {
}

Albany::AdvectiveProblem::~AdvectiveProblem() {
}

void Albany::AdvectiveProblem::buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr) {
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::AdvectiveProblem::buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {
}

void Albany::AdvectiveProblem::constructDirichletEvaluators(
    const std::vector<std::string>& nodeSetIDs) {
}

void Albany::AdvectiveProblem::constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs) {
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::AdvectiveProblem::getValidProblemParameters() const {
}
