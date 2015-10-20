//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_Hydrology.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
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

  has_evolution_equation = params->sublist("FELIX Hydrology").get<bool>("Use Evolution Equation",false);

  // Need to allocate the fields in mesh database
  this->requirements.push_back("surface_height");
  this->requirements.push_back("basal_velocity");
  this->requirements.push_back("ice_thickness");
  this->requirements.push_back("surface_water_input");
  this->requirements.push_back("geothermal_flux");

  if (!has_evolution_equation)
    this->requirements.push_back("drainage_sheet_depth");

  // Set the num PDEs for the null space object to pass to ML
  if (has_evolution_equation)
  {
    this->setNumEquations(2);

    dof_names.resize(2);
    dof_names_dot.resize(1);
    resid_names.resize(2);

    dof_names[0] = "Effective Pressure";
    dof_names[1] = "Drainage Sheet Depth";

    dof_names_dot[0] = "Drainage Sheet Depth Dot";

    resid_names[0] = "Residual Elliptic Eqn";
    resid_names[1] = "Residual Evolution Eqp";
  }
  else
  {
    this->setNumEquations(1);

    dof_names.resize(1);
    resid_names.resize(1);

    dof_names[0] = "Effective Pressure";

    resid_names[0] = "Residual";
  }
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
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);

  return *op.tags;
}

void FELIX::Hydrology::constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = dof_names[0];
  if (has_evolution_equation)
    dirichletNames[1] = dof_names[1];

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
  offsets.resize(neq);

  neumannNames[0] = "Effective Pressure";
  if (has_evolution_equation)
    neumannNames[1] = "Drainage Sheet Depth";
  neumannNames[neq] = "all";

  offsets[0].resize(1);
  offsets[0][0] = 0;
  if (has_evolution_equation)
  {
    offsets[1].resize(1);
    offsets[1][0] = 1;
  }

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
