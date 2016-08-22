//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_Hydrology.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>

FELIX::Hydrology::Hydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
                             const Teuchos::RCP<ParamLib>& paramLib,
                             const int numDimensions) :
  Albany::AbstractProblem (params, paramLib,1),
  numDim (numDimensions)
{
  TEUCHOS_TEST_FOR_EXCEPTION (numDim!=1 && numDim!=2,std::logic_error,"Problem supports only 1D and 2D");

  has_h_equation = params->sublist("FELIX Hydrology").get<bool>("Use Water Thickness Equation",false);
  std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Transient")
    unsteady = true;
  else
    unsteady = false;

  TEUCHOS_TEST_FOR_EXCEPTION (unsteady && !has_h_equation, std::logic_error,
                              "Error! Unsteady case require to use the water thickness equation.\n");

  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }

  // Set the num PDEs for the null space object to pass to ML
  if (has_h_equation)
  {
    this->setNumEquations(2);

    dof_names.resize(2);
    resid_names.resize(2);

    dof_names[0] = "Hydraulic Potential";
    dof_names[1] = "Water Thickness";

    if (unsteady)
    {
      dof_names_dot.resize(1);
      dof_names_dot[0] = "Water Thickness Dot";
    }

    resid_names[0] = "Residual Potential Eqn";
    resid_names[1] = "Residual Thickness Eqp";
  }
  else
  {
    this->setNumEquations(1);

    dof_names.resize(1);
    resid_names.resize(1);

    dof_names[0] = "Hydraulic Potential";

    resid_names[0] = "Residual Potential Eqn";
  }
}

FELIX::Hydrology::~Hydrology()
{
  // Nothing to be done here
}

void FELIX::Hydrology::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                     Albany::StateManager& stateMgr)
{
  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  intrepidBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = Teuchos::rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
  cubature = cubFactory.create(*cellType, meshSpecs[0]->cubatureDegree);

  elementBlockName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = intrepidBasis->getCardinality();
  const int numCellQPs      = cubature->getNumPoints();

  dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim));

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);

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
  if (has_h_equation)
    dirichletNames[1] = dof_names[1];

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);

  // Add the hydrology specific evaluators
  HydrologyDirOp op(*this);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  offsets_ = dirUtils.getOffsets();
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

  neumannNames[0] = "Hydraulic Potential";
  if (has_h_equation)
    neumannNames[1] = "Water Thickness";
  neumannNames[neq] = "all";

  offsets[0].resize(1);
  offsets[0][0] = 0;
  if (has_h_equation)
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
  validPL->sublist("FELIX Field Norm", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");
  validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");

  return validPL;
}
