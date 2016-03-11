//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_Hydrology.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>


FELIX::Hydrology::Hydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
                             const Teuchos::RCP<ParamLib>& paramLib,
                             const int numDimensions) :
  Albany::AbstractProblem (params, paramLib,1),
  numDim (numDimensions)
{
  TEUCHOS_TEST_FOR_EXCEPTION (numDim!=1 && numDim!=2,std::logic_error,"Problem supports only 1D and 2D");

  // Set the num PDEs for the null space object to pass to ML
  this->setNumEquations(1);

  // Need to allocate a fields in mesh database
  this->requirements.push_back("surface_height");
  this->requirements.push_back("basal_friction");
  this->requirements.push_back("sliding_velocity");
  this->requirements.push_back("drainage_sheet_depth");
  this->requirements.push_back("ice_thickness");
  this->requirements.push_back("ice_viscosity");
  this->requirements.push_back("surface_water_input");
  this->requirements.push_back("geothermal_flux");
//  this->requirements.push_back("effective_pressure");

  dof_names.resize(1);
  resid_names.resize(1);
  dof_names[0] = "Hydraulic Potential";
  resid_names[0] = "Hydrology Residual";
}

FELIX::Hydrology::~Hydrology()
{
  // Nothing to be done here
}

void FELIX::Hydrology::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                     Albany::StateManager& stateMgr)
{
  TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  // Build evaluators
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, Teuchos::null);

  // Build bc evaluators
  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present
    constructDirichletEvaluators(*meshSpecs[0]);
  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present
    constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::Hydrology::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                   const Albany::MeshSpecsStruct& meshSpecs,
                                   Albany::StateManager& stateMgr,
                                   Albany::FieldManagerChoice fmchoice,
                                   const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<Hydrology> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void FELIX::Hydrology::constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = dof_names[0];

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
}

void FELIX::Hydrology::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

  // Check to make sure that Neumann BCs are given in the input file
  if (!nbcUtils.haveBCSpecified(this->params))
  {
     return;
  }

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important

  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(1);

  neumannNames[0] = "Hydraulic Potential";
  neumannNames[1] = "all";
  offsets[0].resize(1);
  offsets[0][0] = 0;

  // Construct BC evaluators for all possible names of conditions
  std::vector<std::string> condNames(1);
  condNames[0] = "neumann";

  nfm.resize(1); // FELIX problem only has one element block

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, false, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::Hydrology::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidHydrologyProblemParams");

  validPL->sublist("FELIX Hydrology", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");

  return validPL;
}
