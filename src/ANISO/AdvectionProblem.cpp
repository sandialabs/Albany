//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ANISO_Expression.hpp"
#include "AdvectionProblem.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"

Albany::AdvectionProblem::AdvectionProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<ParamLib>& param_lib,
    const int num_dims,
 	  Teuchos::RCP<const Teuchos::Comm<int> >& commT) :
  Albany::AbstractProblem(params_, param_lib),
  num_dims_(num_dims) {

    std::string fname = params->get<std::string>("MaterialDB Filename");
    material_db_ = Teuchos::rcp(new QCAD::MaterialDatabase(fname, commT));
    this->setNumEquations(1);
    ANISO::expression_init();
}

Albany::AdvectionProblem::~AdvectionProblem() {
}

void Albany::AdvectionProblem::buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr) {

  int phys_sets = meshSpecs.size();
  *out << "Num MeshSpecs: " << phys_sets << std::endl;
  fm.resize(phys_sets);

  for (int ps=0; ps < phys_sets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(
        *fm[ps],
        *meshSpecs[ps],
        stateMgr,
        BUILD_RESID_FM,
        Teuchos::null);
  }

  if (meshSpecs[0]->nsNames.size() > 0)
    constructDirichletEvaluators(meshSpecs[0]->nsNames);

  if (meshSpecs[0]->ssNames.size() > 0)
    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::AdvectionProblem::buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {

  ConstructEvaluatorsOp<AdvectionProblem> op(
      *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void Albany::AdvectionProblem::constructDirichletEvaluators(
    const std::vector<std::string>& nodeSetIDs) {

  std::vector<std::string> bcNames(neq);
  bcNames[0] = "Phi";
  Albany::BCUtils<Albany::DirichletTraits> bcUtils;
  dfm = bcUtils.constructBCEvaluators(
      nodeSetIDs, bcNames, this->params, this->paramLib);
  offsets_ = bcUtils.getOffsets();
}

void Albany::AdvectionProblem::constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs) {

  *out << "unsupported neumann boundary conditions\n";
  abort();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::AdvectionProblem::getValidProblemParameters() const {

  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidAdvectionProblemParams");
  validPL->set<std::string>(
      "MaterialDB Filename", "materials.xml", "XML file name");
  return validPL;
}
