//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_StokesFOHydrology.hpp"

#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>


FELIX::StokesFOHydrology::
StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  numDim(numDim_)
{
  stokes_dof_names.resize(1);
  stokes_resid_names.resize(1);
  stokes_dof_names[0] = "Velocity";
  stokes_resid_names[0] = "Residual Stokes";
  vecDimStokesFO = 2;

  has_evolution_equation = params->sublist("FELIX Hydrology").get<bool>("Use Evolution Equation",false);
  if (has_evolution_equation)
  {
    this->setNumEquations(4);

    hydro_dof_names.resize(2);
    hydro_dof_names_dot.resize(1);
    hydro_resid_names.resize(2);

    hydro_dof_names[0] = "Effective Pressure";
    hydro_dof_names[1] = "Drainage Sheet Depth";

    hydro_dof_names_dot[0] = "Drainage Sheet Depth Dot";

    hydro_resid_names[1] = "Residual Hydrology Elliptic Eqn";
    hydro_resid_names[2] = "Residual Hydrology Evolution Eqp";
  }
  else
  {
    this->setNumEquations(3);

    hydro_dof_names.resize(1);
    hydro_resid_names.resize(2);

    hydro_dof_names[0] = "Effective Pressure";
    hydro_resid_names[1] = "Residual Hydrology Elliptic Eqn";
  }

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);

  // Need to allocate a fields in mesh database
  Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
  for (int i(0); i<req.size(); ++i)
    this->requirements.push_back(req[i]);

  basalSideName   = params->isParameter("Basal Side Name")   ? params->get<std::string>("Basal Side Name")   : "INVALID";
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : "INVALID";
  basalEBName = "INVALID";
  surfaceEBName = "INVALID";
  TEUCHOS_TEST_FOR_EXCEPTION (basalSideName=="INVALID", std::logic_error, "Error! You need to provide a valid 'Basal Side Name',\n" );

  // Need to allocate a fields in basal mesh database
  Teuchos::Array<std::string> breq = params->get<Teuchos::Array<std::string> > ("Required Basal Fields");
  this->ss_requirements[basalSideName].reserve(breq.size()); // Note: this is not for performance, but to guarantee
  for (int i(0); i<breq.size(); ++i)                         //       that ss_requirements.at(basalSideName) does not
    this->ss_requirements[basalSideName].push_back(breq[i]); //       throw, even if it's empty...
  basalReq = this->ss_requirements[basalSideName];

  if (params->isParameter("Required Surface Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="INVALID", std::logic_error, "Error! In order to specify surface requirements, you must also specify a valid 'Surface Side Name'.\n");

    Teuchos::Array<std::string> sreq = params->get<Teuchos::Array<std::string> > ("Required Surface Fields");
    this->ss_requirements[surfaceSideName].reserve(sreq.size()); // Note: same motivation as for the basal side
    for (int i(0); i<sreq.size(); ++i)
      this->ss_requirements[surfaceSideName].push_back(sreq[i]);
  }
}

FELIX::StokesFOHydrology::
~StokesFOHydrology()
{
  // Nothing to be done here
}

void FELIX::StokesFOHydrology::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                             Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);

  elementBlockName = meshSpecs[0]->ebName;

  TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                              "Error! Either 'Basal Side Name' is wrong or something went wrong while building the side mesh specs.\n");
  const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int vecDim          = 2;
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDim));

  // Building also basal side structures
  const CellTopologyData * const basal_side_top = &basalMeshSpecs.ctd;
  basalSideBasis = Albany::getIntrepid2Basis(*basal_side_top);
  basalSideType = rcp(new shards::CellTopology (basal_side_top));

  basalEBName   = basalMeshSpecs.ebName;
  basalCubature = cubFactory.create<PHX::Device, RealType, RealType>(*basalSideType, basalMeshSpecs.cubatureDegree);

  int numBasalSideVertices = basalSideType->getNodeCount();
  int numBasalSideNodes    = basalSideBasis->getCardinality();
  int numBasalSideQPs      = basalCubature->getNumPoints();

  dl_basal = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                     numBasalSideQPs,numDim-1,numDim,vecDim,numCellSides));
  dl->side_layouts[basalSideName] = dl_basal;

  int numSurfaceSideVertices = -1;
  int numSurfaceSideNodes    = -1;
  int numSurfaceSideQPs      = -1;

  if (surfaceSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const surface_side_top = &surfaceMeshSpecs.ctd;
    surfaceSideBasis = Albany::getIntrepid2Basis(*surface_side_top);
    surfaceSideType = rcp(new shards::CellTopology (surface_side_top));

    surfaceEBName   = surfaceMeshSpecs.ebName;
    surfaceCubature = cubFactory.create<PHX::Device, RealType, RealType>(*surfaceSideType, surfaceMeshSpecs.cubatureDegree);

    numSurfaceSideVertices = surfaceSideType->getNodeCount();
    numSurfaceSideNodes    = surfaceSideBasis->getCardinality();
    numSurfaceSideQPs      = surfaceCubature->getNumPoints();

    dl_surface = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                         numSurfaceSideQPs,numDim-1,numDim,vecDim,numCellSides));
    dl->side_layouts[surfaceSideName] = dl_surface;
  }

#ifdef OUTPUT_TO_SCREEN
  *out << "Field Dimensions: \n"
       << "  Workset             = " << worksetSize << "\n"
       << "  Vertices            = " << numCellVertices << "\n"
       << "  CellNodes           = " << numCellNodes << "\n"
       << "  CellQuadPts         = " << numCellQPs << "\n"
       << "  Dim                 = " << numDim << "\n"
       << "  VecDim              = " << vecDim << "\n"
       << "  BasalSideVertices   = " << numBasalSideVertices << "\n"
       << "  BasalSideNodes      = " << numBasalSideNodes << "\n"
       << "  BasalSideQuadPts    = " << numBasalQPs << "\n"
       << "  SurfaceSideVertices = " << numSurfaceSideVertices << "\n"
       << "  SurfaceSideNodes    = " << numSurfaceSideNodes << "\n"
       << "  SurfaceSideQuadPts  = " << numSurfaceQPs << std::endl;
#endif

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present
     constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::StokesFOHydrology::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFOHydrology> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
FELIX::StokesFOHydrology::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  for (int i=0; i<neq; i++) {
    std::stringstream s; s << "U" << i;
    dirichletNames[i] = s.str();
  }
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                         this->params, this->paramLib);
}

// Neumann BCs
void FELIX::StokesFOHydrology::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

  // Check to make sure that Neumann BCs are given in the input file

  if(!nbcUtils.haveBCSpecified(this->params)) {
     return;
  }


  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important

  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq + 1);

  neumannNames[0] = "U0";
  offsets[0].resize(1);
  offsets[0][0] = 0;
  offsets[neq].resize(neq);
  offsets[neq][0] = 0;

  neumannNames[1] = "U1";
  offsets[1].resize(1);
  offsets[1][0] = 1;
  offsets[neq][1] = 1;

  neumannNames[2] = "Phi";
  offsets[2].resize(1);
  offsets[2][0] = 2;
  offsets[neq][2] = 2;

  if (has_evolution_equation)
  {
    neumannNames[3] = "N";
    offsets[3].resize(1);
    offsets[3][0] = 2;
    offsets[neq][3] = 3;
  }

  neumannNames[neq] = "all";

  // Construct BC evaluators for all possible names of conditions
  std::vector<std::string> condNames(2);
  condNames[0] = "lateral";
  condNames[1] = "neumann";

  nfm.resize(1); // FELIX problem only has one element block

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, stokes_dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::StokesFOHydrology::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesFOHydrologyProblemParams");

  validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Stereographic Map", false, "");
  validPL->sublist("FELIX Hydrology", false, "");
  validPL->sublist("FELIX Viscosity", false, "");
  validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
  validPL->sublist("FELIX Surface Gradient", false, "");
  validPL->sublist("Equation Set", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");
  validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");

  return validPL;
}
