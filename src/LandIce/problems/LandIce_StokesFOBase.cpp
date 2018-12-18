#include "LandIce_StokesFOBase.hpp"
#include "Teuchos_CompilerCodeTweakMacros.hpp"

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
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : INVALID_STR;

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  basalSideName = params->isParameter("Basal Side Name") ? params->get<std::string>("Basal Side Name") : INVALID_STR;

  // Setup velocity dof and resid names. Derived classes should _append_ to these
  dof_names.resize(1);
  resid_names.resize(1);
  scatter_names.resize(1);

  dof_names[0] = "Velocity";
  resid_names[0] = dof_names[0] + " Residual";
  scatter_names[0] = "Scatter " + resid_names[0];

  offsetVelocity = 0;
  vecDimFO       = std::min((int)neq,(int)2);

  // Names of some common fields. User can set them in the problem section, in case they are
  // loaded from mesh, where they are saved with a different name
  surface_height_name    = params->get<std::string>("Surface Height Name","surface_height");
  ice_thickness_name     = params->get<std::string>("Ice Thickness Name" ,"ice_thickness");
  temperature_name       = params->get<std::string>("Temperature Name"   ,"temperature");
  bed_topography_name    = params->get<std::string>("Bed Topography Name","bed_topography");
  flow_factor_name       = params->get<std::string>("Flow Factor Name"      , "flow_factor");
  stiffening_factor_name = params->get<std::string>("Stiffening Factor Name", "stiffening_factor");
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
  if (!isInvalid(surfaceSideName) && dl->side_layouts.find(surfaceSideName)==dl->side_layouts.end())
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

  // If we have thickness or surface velocity diagnostics, we may need basal side stuff
  if (!isInvalid(basalSideName) && dl->side_layouts.find(basalSideName)==dl->side_layouts.end())
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

  // Prepare the requests of interpolation/utility evaluators
  setupEvaluatorRequests ();

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
StokesFOBase::getStokesFOBaseProblemParameters () const
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
  validPL->set<std::string> ("Surface Height Name"   , "", "Name of the surface height field");
  validPL->set<std::string> ("Ice Thickness Name"    , "", "Name of the ice thickness field");
  validPL->set<std::string> ("Temperature Name"      , "", "Name of the temperature field");
  validPL->set<std::string> ("Bed Topography Name"   , "", "Name of the bed topography field");
  validPL->set<std::string> ("Flow Factor Name"      , "", "Name of the flow factor field");
  validPL->set<std::string> ("Stiffening Factor Name", "", "Name of the stiffening factor field");

  return validPL;
}

void StokesFOBase::requestInterpolationEvaluator (
    const std::string& fname,
    const int rank,
    const FieldLocation location,
    const FieldScalarType scalar_type,
    const InterpolationRequest request)
{
  TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)!=field_rank.end() && field_rank[fname]!=rank, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " with rank " + std::to_string(rank) +
                              ", when it was previously marked as of rank " + std::to_string(field_rank[fname]) + ".\n");
  TEUCHOS_TEST_FOR_EXCEPTION (field_location.find(fname)!=field_location.end() && field_location[fname]!=location, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " as located at " + e2str(location) +
                              ", when it was previously marked as located at " + e2str(field_location[fname]) + ".\n");

  build_interp_ev[fname][request] = true;
  field_rank[fname] = rank;
  field_location[fname] = location;
  field_scalar_type[fname] |= scalar_type;
}

void StokesFOBase::requestSideSetInterpolationEvaluator (
    const std::string& ss_name,
    const std::string& fname,
    const int rank,
    const FieldLocation location,
    const FieldScalarType scalar_type,
    const InterpolationRequest request)
{
  TEUCHOS_TEST_FOR_EXCEPTION (ss_field_rank[ss_name].find(fname)!=ss_field_rank[ss_name].end() && ss_field_rank[ss_name][fname]!=rank, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " with rank " + std::to_string(rank) +
                              ", when it was previously marked as of rank " + std::to_string(ss_field_rank[ss_name][fname]) + ".\n");
  TEUCHOS_TEST_FOR_EXCEPTION (ss_field_location[ss_name].find(fname)!=ss_field_location[ss_name].end() && ss_field_location[ss_name][fname]!=location, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " as located at " + e2str(location) +
                              ", when it was previously marked as located at " + e2str(ss_field_location[ss_name][fname]) + ".\n");

  ss_build_interp_ev[ss_name][fname][request] = true;
  ss_field_rank[ss_name][fname] = rank;
  ss_field_location[ss_name][fname] = location;
  ss_field_scalar_type[ss_name][fname] |= scalar_type;
}

void StokesFOBase::setupEvaluatorRequests ()
{
  // Note: for all fields except dof_names[0], we are assuming that the scalar type is ParamScalar.
  //       It is up to the derived problems to adjust this if that's not the case (e.g., ice_thickness
  //       is a MeshScalar in StokesFOThickness).

  // Volume required interpolations
  requestInterpolationEvaluator(dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::QP_VAL); 
  requestInterpolationEvaluator(dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::GRAD_QP_VAL); 
  requestInterpolationEvaluator(surface_height_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
#ifndef CISM_HAS_LANDICE
  // If not coupled with cism, we may have to compute the surface gradient ourselves
  requestInterpolationEvaluator(surface_height_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::GRAD_QP_VAL); 
#endif
  requestInterpolationEvaluator(temperature_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_VAL); 
  requestInterpolationEvaluator(stiffening_factor_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 

  // Basal Friction BC requests
  for (auto it : landice_bcs[LandIceBC::BasalFriction]) {
    std::string ssName = it->get<std::string>("Side Set Name");
    // BasalFriction BC needs velocity on the side, and BFs/coords
    // And if we compute grad beta, we may even need the velocity gradient and the effective pressure gradient
    // (which, if effevtive_pressure is a dist param, needs to be projected to the side)

    ss_utils_needed[ssName][UtilityRequest::BFS] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;  // Only really needed if stereographic map is used.

    requestSideSetInterpolationEvaluator(ssName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(ssName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(ssName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::GRAD_QP_VAL); 
    requestSideSetInterpolationEvaluator(ssName, "effective_pressure", 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::GRAD_QP_VAL); 
    requestSideSetInterpolationEvaluator(ssName, "effective_pressure", 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE_IF_DIST_PARAM); 

    // For "Given Field" and "Exponent of Given Field" we also need to interpolate the given field at the quadrature points
    auto& bfc = it->sublist("Basal Friction Coefficient");
    const auto type = util::upper_case(bfc.get<std::string>("Type"));
    if (type=="GIVEN FIELD" || type=="EXPONENT OF GIVEN FIELD") {
      requestSideSetInterpolationEvaluator(ssName, bfc.get<std::string>("Given Field Variable Name"), 0,
                                           FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL);
      requestSideSetInterpolationEvaluator(ssName, bfc.get<std::string>("Given Field Variable Name"), 0,
                                           FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE_IF_DIST_PARAM);
    }

    // If zero on floating, we also need bed topography and thickness
    if (bfc.get<bool>("Zero Beta On Floating Ice", false)) {
      requestSideSetInterpolationEvaluator(ssName, bed_topography_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL);
      requestSideSetInterpolationEvaluator(ssName, bed_topography_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE_IF_DIST_PARAM);

      requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL);
      requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE_IF_DIST_PARAM);
    }
  }

  // Lateral BC requests
  for (auto it : landice_bcs[LandIceBC::Lateral]) {
    std::string ssName = it->get<std::string>("Side Set Name");

    // Lateral bc needs thickness ...
    requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 

    // ... possibly surface height ...
    requestSideSetInterpolationEvaluator(ssName, surface_height_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(ssName, surface_height_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 

    // ... and BFs (including normals)
    ss_utils_needed[ssName][UtilityRequest::BFS] = true;
    ss_utils_needed[ssName][UtilityRequest::NORMALS] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;
  }

  // Surface diagnostics
  if (!isInvalid(surfaceSideName)) {
    // Surface velocity diagnostic requires dof at qps on surface side
    requestSideSetInterpolationEvaluator(surfaceSideName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(surfaceSideName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::QP_VAL); 

    // ... and observed surface velocity ...
    // NOTE: RMS could be either scalar or vector. The states registration should have figure it out (if the field is listed as input).
    requestSideSetInterpolationEvaluator(surfaceSideName, "observed_surface_velocity", 1, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
    const int obs_vel_rms_rank = ss_field_rank[surfaceSideName]["observed_surface_velocity_RMS"];
    requestSideSetInterpolationEvaluator(surfaceSideName, "observed_surface_velocity_RMS", obs_vel_rms_rank, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 

    // ... and BFs
    ss_utils_needed[surfaceSideName][UtilityRequest::BFS] = true;

    if (!isInvalid(basalSideName)) {
      // Surface velocity diagnostics *may* add a basal side regularization
      requestSideSetInterpolationEvaluator(basalSideName, stiffening_factor_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE_IF_DIST_PARAM); 
      requestSideSetInterpolationEvaluator(basalSideName, stiffening_factor_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
      requestSideSetInterpolationEvaluator(basalSideName, stiffening_factor_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::GRAD_QP_VAL); 

      ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;
    }
  }

  // SMB-related diagnostics
  if (!isInvalid(basalSideName)) {
    // Needs BFs
    ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;

    // SMB evaluators may need velocity, averaged velocity, and thickness
    requestSideSetInterpolationEvaluator(basalSideName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(basalSideName, dof_names[0], 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, "averaged_velocity", 1, FieldLocation::Node, FieldScalarType::Scalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, "surface_mass_balance", 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, "surface_mass_balance_RMS", 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::GRAD_QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, ice_thickness_name, 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::CELL_TO_SIDE_IF_DIST_PARAM); 
    requestSideSetInterpolationEvaluator(basalSideName, "observed_ice_thickness", 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, "observed_ice_thickness_RMS", 0, FieldLocation::Node, FieldScalarType::ParamScalar, InterpolationRequest::QP_VAL); 
  }
}

} // namespace LandIce
