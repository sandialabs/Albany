//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "FELIX_StokesFOThermoCoupled.hpp"

FELIX::StokesFOThermoCoupled::
StokesFOThermoCoupled( const Teuchos::RCP<Teuchos::ParameterList>& params_,
                       const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                       const Teuchos::RCP<ParamLib>& paramLib_,
                       const int numDim_) :
                       Albany::AbstractProblem(params_, paramLib_, numDim_),
                       numDim(numDim_),
                       discParams(discParams_)
{
  // 2 eqns for Stokes FO + 1 eqn. for enthalpy + 1 eqn. for w_z
  this->setNumEquations(4);

  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }

  basalSideName   = params->isParameter("Basal Side Name")   ? params->get<std::string>("Basal Side Name")   : "INVALID";
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : "INVALID";
  basalEBName = "INVALID";
  surfaceEBName = "INVALID";
  sliding = params->isSublist("FELIX Basal Friction Coefficient");
  TEUCHOS_TEST_FOR_EXCEPTION (sliding && basalSideName=="INVALID", std::logic_error,
                              "Error! With sliding, you need to provide a valid 'Basal Side Name',\n" );

  if (params->isParameter("Required Basal Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (basalSideName=="INVALID", std::logic_error, "Error! In order to specify basal requirements, you must also specify a valid 'Basal Side Name'.\n");

    // Need to allocate a fields in basal mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Basal Fields");
    this->ss_requirements[basalSideName].reserve(req.size()); // Note: this is not for performance, but to guarantee
    for (int i(0); i<req.size(); ++i)                         //       that ss_requirements.at(basalSideName) does not
      this->ss_requirements[basalSideName].push_back(req[i]); //       throw, even if it's empty...
  }
  if (params->isParameter("Required Surface Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="INVALID", std::logic_error, "Error! In order to specify surface requirements, you must also specify a valid 'Surface Side Name'.\n");

    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Surface Fields");
    this->ss_requirements[surfaceSideName].reserve(req.size()); // Note: same motivation as for the basal side
    for (int i(0); i<req.size(); ++i)
      this->ss_requirements[surfaceSideName].push_back(req[i]);
  }

  //Teuchos::ParameterList SUPG_list = params->get<Teuchos::ParameterList>("SUPG Settings");
  //haveSUPG = SUPG_list.get("Have SUPG Stabilization",false);
  needsDiss = params->get<bool> ("Needs Dissipation",true);
  needsBasFric = params->get<bool> ("Needs Basal Friction",true);
  isGeoFluxConst = params->get<bool> ("Constant Geothermal Flux",true);
}

FELIX::StokesFOThermoCoupled::
~StokesFOThermoCoupled()
{
  // Nothing to be done here
}

void FELIX::StokesFOThermoCoupled::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
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

  const int worksetSize     = meshSpecs[0]->worksetSize;
  vecDimFO                  = std::min((int)neq,(int)2);
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDimFO));
  dl_scalar = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim, 1));

  int numSurfaceSideVertices = -1;
  int numSurfaceSideNodes    = -1;
  int numSurfaceSideQPs      = -1;
  int numBasalSideVertices   = -1;
  int numBasalSideNodes      = -1;
  int numBasalSideQPs        = -1;

  if (basalSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                "Error! Either 'Basal Side Name' is wrong or something went wrong while building the side mesh specs.\n");
    const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

    // Building also basal side structures
    const CellTopologyData * const side_top = &basalMeshSpecs.ctd;
    basalSideBasis = Albany::getIntrepid2Basis(*side_top);
    basalSideType = rcp(new shards::CellTopology (side_top));

    basalEBName   = basalMeshSpecs.ebName;
    basalCubature = cubFactory.create<PHX::Device, RealType, RealType>(*basalSideType, basalMeshSpecs.cubatureDegree);

    numBasalSideVertices = basalSideType->getNodeCount();
    numBasalSideNodes    = basalSideBasis->getCardinality();
    numBasalSideQPs      = basalCubature->getNumPoints();

    dl_basal = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                       numBasalSideQPs,numDim-1,numDim,numCellSides,neq));
    dl_side_scalar = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                             numBasalSideQPs,numDim-1,numDim,numCellSides,1));
    dl->side_layouts[basalSideName] = dl_basal;

    dl_scalar->side_layouts[basalSideName] = dl_side_scalar;
  }

  if (surfaceSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const side_top = &surfaceMeshSpecs.ctd;
    surfaceSideBasis = Albany::getIntrepid2Basis(*side_top);
    surfaceSideType = rcp(new shards::CellTopology (side_top));

    surfaceEBName   = surfaceMeshSpecs.ebName;
    surfaceCubature = cubFactory.create<PHX::Device, RealType, RealType>(*surfaceSideType, surfaceMeshSpecs.cubatureDegree);

    numSurfaceSideVertices = surfaceSideType->getNodeCount();
    numSurfaceSideNodes    = surfaceSideBasis->getCardinality();
    numSurfaceSideQPs      = surfaceCubature->getNumPoints();

    dl_surface = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                         numSurfaceSideQPs,numDim-1,numDim,numCellSides,vecDimFO));
    dl->side_layouts[surfaceSideName] = dl_surface;
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  int commRank = Teuchos::GlobalMPISession::getRank();
  int commSize = Teuchos::GlobalMPISession::getNProc();
  out->setProcRankAndSize(commRank, commSize);
  out->setOutputToRootOnly(0);

  *out << "Field Dimensions: \n"
      << "  Workset             = " << worksetSize << "\n"
      << "  Vertices            = " << numCellVertices << "\n"
      << "  CellNodes           = " << numCellNodes << "\n"
      << "  CellQuadPts         = " << numCellQPs << "\n"
      << "  Dim                 = " << numDim << "\n"
      << "  VecDim              = " << neq << "\n"
      << "  VecDimFO            = " << vecDimFO << "\n"
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
FELIX::StokesFOThermoCoupled::
buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFOThermoCoupled> op(
      *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
FELIX::StokesFOThermoCoupled::constructDirichletEvaluators(
    const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  for (int i=0; i<vecDimFO; i++) {
    std::stringstream s; s << "U" << i;
    dirichletNames[i] = s.str();
  }
  dirichletNames[vecDimFO] = "W";
  dirichletNames[vecDimFO+1] = "Enth";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
  offsets_ = dirUtils.getOffsets();
}

// Neumann BCs
void FELIX::StokesFOThermoCoupled::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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

  int neq_Stokes = 2;
  std::vector<std::string> neumannNames(neq_Stokes + 1);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq_Stokes + 1);

  neumannNames[0] = "U0";
  offsets[0].resize(1);
  offsets[0][0] = 0;
  offsets[neq_Stokes].resize(neq_Stokes);
  offsets[neq_Stokes][0] = 0;
  neumannNames[1] = "U1";
  offsets[1].resize(1);
  offsets[1][0] = 1;
  offsets[neq_Stokes][1] = 1;
  /*
	neumannNames[2] = "W";
	offsets[2].resize(1);
	offsets[2][0] = 2;
	offsets[neq][0] = 2;

	neumannNames[3] = "Enth";
	offsets[3].resize(1);
	offsets[3][0] = 3;
	offsets[neq][0] = 3;
   */

  neumannNames[neq_Stokes] = "Stokes";

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
  std::vector<std::string> condNames(6); //(dCdx, dCdy, dCdz), dCdn, basal, P, lateral, basal_scalar_field
  Teuchos::ArrayRCP<std::string> dof_names(1);
  dof_names[0] = "Velocity";

  // Note that sidesets are only supported for two and 3D currently
  if(numDim == 2)
    condNames[0] = "(dFluxdx, dFluxdy)";
  else if(numDim == 3)
    condNames[0] = "(dFluxdx, dFluxdy, dFluxdz)";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

  condNames[1] = "dFluxdn";
  condNames[2] = "basal";
  condNames[3] = "P";
  condNames[4] = "lateral";
  condNames[5] = "basal_scalar_field";

  nfm.resize(1); // FELIX problem only has one element block

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::StokesFOThermoCoupled::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getGenericProblemParams("ValidStokesFOProblemParams");

  validPL->set<bool> ("Extruded Column Coupled in 2D Response", false, "Boolean describing whether the extruded column is coupled in 2D response");
  validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");
  validPL->set<int> ("importCellTemperatureFromMesh", 0, "");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Stereographic Map", false, "");
  validPL->sublist("FELIX Viscosity", false, "");
  validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
  validPL->sublist("FELIX L2 Projected Boundary Laplacian", false, "Parameters needed to compute the L2 Projected Boundary Laplacian");
  validPL->sublist("FELIX Surface Gradient", false, "");
  validPL->sublist("Equation Set", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->set<bool> ("Needs Dissipation", true, "Boolean describing whether we take into account the heat generated by dissipation");
  validPL->set<bool> ("Needs Basal Friction", true, "Boolean describing whether we take into account the heat generated by basal friction");
  validPL->set<bool> ("Constant Geothermal Flux", true, "Boolean describing whether the geothermal flux is constant");
  validPL->sublist("FELIX Enthalpy Stabilization", false, "Stabilization used for Enthalpy equation. Options: Streamline Upwind, SUPG, None");
  validPL->sublist("FELIX Field Norm", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");
  validPL->sublist("FELIX Noise", false, "");
  validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");
  validPL->set<bool>("Use Time Parameter", false, "Solely to use Solver Method = Continuation");
  validPL->set<bool>("Print Stress Tensor", false, "Whether to save stress tensor in the mesh");

  return validPL;
}
