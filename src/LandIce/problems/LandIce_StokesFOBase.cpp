#include "LandIce_StokesFOBase.hpp"

#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

namespace LandIce {

StokesFOBase::
StokesFOBase (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                        const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                        const Teuchos::RCP<ParamLib>& paramLib_,
                        const int numDim_)
 : Albany::AbstractProblem(params_, paramLib_, numDim_)
 , discParams (discParams_)
 , numDim(numDim_)
 , use_sdbcs_(false)
{
  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }

  // Parsing the LandIce boundary conditions sublist
  auto landice_bcs_params = Teuchos::sublist(params,"LandIce BCs");
  int num_bcs = landice_bcs_params->get<int>("Number",0);
  for (int i=0; i<num_bcs; ++i) {
    auto this_bc = Teuchos::sublist(landice_bcs_params,Albany::strint("BC",i));
    std::string type_str = util::upper_case(this_bc->get<std::string>("Type"));

    LandIceBC type;
    if (type_str=="BASAL FRICTION") {
      type = LandIceBC::BasalFriction;
    } else if (type_str=="LATERAL") {
      type = LandIceBC::Lateral;
    } else if (type_str=="SYNTETIC TEST") {
      type = LandIceBC::SynteticTest;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Unknown LandIce bc '" + type_str + "'.\n");
    }
    landice_bcs[type].push_back(this_bc);
  }

  // Surface side, where velocity diagnostics are computed (e.g., velocity mismatch)
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : "__INVALID__";

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  basalSideName = params->isParameter("Basal Side Name") ? params->get<std::string>("Basal Side Name") : "__INVALID__";

  // Setup velocity dof and resid names. Derived classes should _append_ to these
  dof_names.resize(1);
  resid_names.resize(1);
  scatter_names.resize(1);

  dof_names[0] = "Velocity";
  resid_names[0] = dof_names[0] + " Residual";
  scatter_names[0] = "Scatter " + resid_names[0];

  offsetVelocity = 0;
  vecDimFO       = std::min((int)neq,(int)2);
}

void StokesFOBase::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                 Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellType->getSideCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDimFO));

  int sideDim = numDim-1;
  for (auto it : landice_bcs) {
    for (auto pl: it.second) {
      std::string ssName = pl->get<std::string>("Side Set Name");
      TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(ssName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either the side set name is wrong or something went wrong while building the side mesh specs.\n");
      const Albany::MeshSpecsStruct& sideMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ssName)[0];

      // Building also side structures
      const CellTopologyData * const side_top = &sideMeshSpecs.ctd;
      sideBasis[ssName] = Albany::getIntrepid2Basis(*side_top);
      sideType[ssName] = rcp(new shards::CellTopology (side_top));

      // If there's no side discretiation, then sideMeshSpecs.cubatureDegree will be -1, and the user need to specify a cubature degree somewhere else
      int sideCubDegree = sideMeshSpecs.cubatureDegree;
      if (pl->isParameter("Cubature Degree")) {
        sideCubDegree = pl->get<int>("Cubature Degree");
      }
      TEUCHOS_TEST_FOR_EXCEPTION (sideCubDegree<0, std::runtime_error, "Error! Missing cubature degree information on side '" << ssName << ".\n"
                                                                       "       Either add a side discretization, or specify 'Cubature Degree' in sublist '" + pl->name() + "'.\n");
      sideCubature[ssName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[ssName], sideCubDegree);

      int numSideVertices = sideType[ssName]->getNodeCount();
      int numSideNodes    = sideBasis[ssName]->getCardinality();
      int numSideQPs      = sideCubature[ssName]->getNumPoints();

      dl->side_layouts[ssName] = rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,
                                                         numSideQPs,sideDim,numDim,numCellSides,vecDimFO));
    }
  }

  // If we have velocity diagnostics, we need surface side stuff
  if (surfaceSideName!="__INVALID__" && dl->side_layouts.find(surfaceSideName)==dl->side_layouts.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const side_top = &surfaceMeshSpecs.ctd;
    sideBasis[surfaceSideName] = Albany::getIntrepid2Basis(*side_top);
    sideType[surfaceSideName]= rcp(new shards::CellTopology (side_top));

    sideCubature[surfaceSideName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[surfaceSideName], surfaceMeshSpecs.cubatureDegree);

    int numSurfaceSideVertices = sideType[surfaceSideName]->getNodeCount();
    int numSurfaceSideNodes    = sideBasis[surfaceSideName]->getCardinality();
    int numSurfaceSideQPs      = sideCubature[surfaceSideName]->getNumPoints();

    dl->side_layouts[surfaceSideName] = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                                                numSurfaceSideQPs,sideDim,numDim,numCellSides,vecDimFO));
  }

  // If we have thickness diagnostics, we need basal side stuff
  if (basalSideName!="__INVALID__" && dl->side_layouts.find(basalSideName)==dl->side_layouts.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either 'Basal Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

    // Building also basal side structures
    const CellTopologyData * const side_top = &basalMeshSpecs.ctd;
    sideBasis[basalSideName] = Albany::getIntrepid2Basis(*side_top);
    sideType[basalSideName]= rcp(new shards::CellTopology (side_top));

    sideCubature[basalSideName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[basalSideName], basalMeshSpecs.cubatureDegree);

    int numbasalSideVertices = sideType[basalSideName]->getNodeCount();
    int numbasalSideNodes    = sideBasis[basalSideName]->getCardinality();
    int numbasalSideQPs      = sideCubature[basalSideName]->getNumPoints();

    dl->side_layouts[basalSideName] = rcp(new Albany::Layouts(worksetSize,numbasalSideVertices,numbasalSideNodes,
                                                              numbasalSideQPs,sideDim,numDim,numCellSides,vecDimFO));
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  int commRank = Teuchos::GlobalMPISession::getRank();
  int commSize = Teuchos::GlobalMPISession::getNProc();
  out->setProcRankAndSize(commRank, commSize);
  out->setOutputToRootOnly(0);

  *out << "=== Field Dimensions ===\n"
       << " Volume:\n"
       << "   Workset     = " << worksetSize << "\n"
       << "   Vertices    = " << numCellVertices << "\n"
       << "   CellNodes   = " << numCellNodes << "\n"
       << "   CellQuadPts = " << numCellQPs << "\n"
       << "   Dim         = " << numDim << "\n"
       << "   VecDim      = " << neq << "\n"
       << "   VecDimFO    = " << vecDimFO << "\n";
  for (auto it : dl_side) {
    *out << " Side Set '" << it.first << "':\n" 
         << "  Vertices   = " << it.second->vertices_vector->dimension(1) << "\n"
         << "  Nodes      = " << it.second->node_scalar->dimension(1) << "\n"
         << "  QuadPts    = " << it.second->qp_scalar->dimension(1) << "\n";
  }
#endif

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);

  // Build a dirichlet fm if nodesets are present
  if (meshSpecs[0]->nsNames.size() >0) {
    constructDirichletEvaluators(*meshSpecs[0]);
  }

  // Build a neumann fm if sidesets are present
  if(meshSpecs[0]->ssNames.size() > 0) {
     constructNeumannEvaluators(meshSpecs[0]);
  }
}

Teuchos::RCP<Teuchos::ParameterList>
LandIce::StokesFOBase::getStokesFOBaseProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidStokesFOProblemParams");

  validPL->set<bool> ("Extruded Column Coupled in 2D Response", false, "Boolean describing whether the extruded column is coupled in 2D response");
  validPL->sublist("Stereographic Map", false, "");
  validPL->sublist("LandIce BCs", false, "Specify boundary conditions specific to LandIce (bypass usual Neumann/Dirichlet classes)");
  validPL->sublist("LandIce Viscosity", false, "");
  validPL->sublist("LandIce Effective Pressure Surrogate", false, "Parameters needed to compute the effective pressure surrogate");
  validPL->sublist("LandIce L2 Projected Boundary Laplacian", false, "Parameters needed to compute the L2 Projected Boundary Laplacian");
  validPL->sublist("LandIce Surface Gradient", false, "");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("LandIce Field Norm", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("LandIce Noise", false, "");
  validPL->set<bool>("Use Time Parameter", false, "Solely to use Solver Method = Continuation");
  validPL->set<bool>("Print Stress Tensor", false, "Whether to save stress tensor in the mesh");
  validPL->set<bool>("Adjust Bed Topography to Account for Thickness Changes", false, "");
  validPL->set<bool>("Adjust Surface Height to Account for Thickness Changes", false, "");

  return validPL;
}

} // namespace LandIce
