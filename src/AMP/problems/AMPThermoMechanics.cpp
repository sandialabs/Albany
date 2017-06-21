//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AMPThermoMechanics.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"


Albany::AMPThermoMechanics::
AMPThermoMechanics(const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<ParamLib>& param_lib,
    const int num_dims,
 	  Teuchos::RCP<const Teuchos::Comm<int> >& commT) :
  Albany::AbstractProblem(params_, param_lib),
  num_dims_(num_dims), hasConsolidation_(true)
{
  // Read the "MaterialDB Filename" parameter from the input deck and create the MaterialDatabase
  std::string filename = params->get<std::string>("MaterialDB Filename");
  material_db_ = Teuchos::rcp(new Albany::MaterialDatabase(filename, commT));

  // get consolidation flag from material input deck. If not specified then assign true.
  hasConsolidation_ = material_db_->getParam<bool>("Compute Consolidation",true);
  
  // Tell user if consolidation is on or off
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  
  if (! hasConsolidation_ ) {
      *out << "*******************************" << std::endl;
      *out << "WARNING: Consolidation is OFF" << std::endl;
      *out << "*******************************" << std::endl;
  }
  
  
  this->setNumEquations(1);
}

Albany::AMPThermoMechanics::
~AMPThermoMechanics()
{ }

// This function return true if compute consolidation was specified in the
// input deck. By default consolidation is on.

bool
Albany::AMPThermoMechanics::hasConsolidation() const
{
  return hasConsolidation_;
}


void Albany::AMPThermoMechanics::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  int phys_sets = meshSpecs.size();
  *out << "Num MeshSpecs: " << phys_sets << std::endl;
  fm.resize(phys_sets);
  
  for (int ps=0; ps<phys_sets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
        Teuchos::null);
  }

  if(meshSpecs[0]->nsNames.size() > 0)
    constructDirichletEvaluators(meshSpecs[0]->nsNames);

  if(meshSpecs[0]->ssNames.size() > 0)
    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::AMPThermoMechanics::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  ConstructEvaluatorsOp<AMPThermoMechanics> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void Albany::AMPThermoMechanics::constructDirichletEvaluators(
    const std::vector<std::string>& nodeSetIDs)
{
  std::vector<std::string> bcNames(neq);
  bcNames[0] = "Temperature";
  Albany::BCUtils<Albany::DirichletTraits> bcUtils;
  dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
      this->params, this->paramLib);
  offsets_ = bcUtils.getOffsets(); 
}

// Neumann BCs
void Albany::AMPThermoMechanics::constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  Albany::BCUtils<Albany::NeumannTraits> bcUtils;

  if(!bcUtils.haveBCSpecified(this->params))
    return;

  std::vector<std::string> bcNames(neq);
  Teuchos::ArrayRCP<std::string> dof_names(neq);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq);

  bcNames[0] = "U";
  dof_names[0] = "u";
  offsets[0].resize(1);
  offsets[0][0] = 0;

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
  std::vector<std::string> condNames(4); 
  //dudx, dudy, dudz, dudn, scaled jump (internal surface), or robin (like DBC plus scaled jump)

  // Note that sidesets are only supported for two and 3D currently
  if(num_dims_ == 2)
    condNames[0] = "(dudx, dudy)";
  else if(num_dims_ == 3)
    condNames[0] = "(dudx, dudy, dudz)";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

  condNames[1] = "dudn";

  condNames[2] = "scaled jump";

  condNames[3] = "robin";

  nfm.resize(1); // Heat problem only has one physics set   
  nfm[0] = bcUtils.constructBCEvaluators(meshSpecs, bcNames, dof_names, false, 0,
      condNames, offsets, dl_, this->params, this->paramLib);

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::AMPThermoMechanics::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidAMPThermoMechanicsParams");

  validPL->set<std::string>("MaterialDB Filename",
                            "materials.xml",
                            "Filename of material database xml file");
  // this is for use in CTM project
  validPL->set<bool>("Transient",
                true,
                "Specify if you want a transient analysis or not");

  return validPL;
}
