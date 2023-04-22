#include "LandIce_StokesFOBase.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "Albany_ProblemUtils.hpp"  // For 'getIntrepidwBasis'
#include "Albany_StringUtils.hpp" // for 'upper_case'

#include "Teuchos_CompilerCodeTweakMacros.hpp"

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
 , params(params_)
{
  // Parsing the LandIce boundary conditions sublist
  auto landice_bcs_params = Teuchos::sublist(params,"LandIce BCs");
  unsigned int num_bcs = landice_bcs_params->get<int>("Number",0);
  for (unsigned int i=0; i<num_bcs; ++i) {
    auto this_bc = Teuchos::sublist(landice_bcs_params,util::strint("BC",i));
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

  Teuchos::ParameterList& physics_list = params->sublist("LandIce Physical Parameters");

  if (physics_list.isParameter("Clausius-Clapeyron Coefficient") &&
      physics_list.get<double>("Clausius-Clapeyron Coefficient")!=0.0) {
    viscosity_use_corrected_temperature = true;
  } else {
    viscosity_use_corrected_temperature = false;
  }

  viscosity_use_p0_temperature = params->sublist("LandIce Viscosity").get("Use P0 Temperature",true);

  compute_dissipation = params->sublist("LandIce Viscosity").get("Extract Strain Rate Sq", false);

  if (!physics_list.isParameter("Atmospheric Pressure Melting Temperature")) {
    physics_list.set("Atmospheric Pressure Melting Temperature", 273.15);
  }

  if (!physics_list.isParameter("Seconds per Year")) {
    physics_list.set("Seconds per Year", 3.1536e7);
  }


  // Setup velocity dof and resid names. Derived classes should _append_ to these
  dof_names.resize(1);
  resid_names.resize(1);
  scatter_names.resize(1);

  velocity_name = dof_names[0] = "Velocity";
  resid_names[0] = dof_names[0] + " Residual";
  scatter_names[0] = "Scatter " + resid_names[0];

  dof_offsets.resize(1);
  dof_offsets[0] = 0;
  vecDimFO = std::min((int)neq,(int)2);

  // Names of some common fields. User can set them in the problem section, in case they are
  // loaded from mesh, where they are saved with a different name
  body_force_name             = params->sublist("Variables Names").get<std::string>("Body Force Name","body_force");
  surface_height_name         = params->sublist("Variables Names").get<std::string>("Surface Height Name","surface_height");
  surface_height_param_name   = surface_height_name + "_param";
  surface_height_observed_name= "observed_" +  surface_height_name;
  ice_thickness_name          = params->sublist("Variables Names").get<std::string>("Ice Thickness Name" ,"ice_thickness");
  temperature_name            = params->sublist("Variables Names").get<std::string>("Temperature Name"   ,"temperature");
  corrected_temperature_name  = "corrected_" + temperature_name;
  bed_topography_name         = params->sublist("Variables Names").get<std::string>("Bed Topography Name"    ,"bed_topography");
  bed_topography_param_name   = bed_topography_name + "_param";
  bed_topography_observed_name= "observed_" + bed_topography_name;
  flow_factor_name            = params->sublist("Variables Names").get<std::string>("Flow Factor Name"       ,"flow_factor");
  stiffening_factor_name      = params->sublist("Variables Names").get<std::string>("Stiffening Factor Name" ,"stiffening_factor");
  effective_pressure_name     = params->sublist("Variables Names").get<std::string>("Effective Pressure Name","effective_pressure");
  vertically_averaged_velocity_name = params->sublist("Variables Names").get<std::string>("Vertically Averaged Velocity Name","vertically_averaged_velocity");
  flux_divergence_name        = params->sublist("Variables Names").get<std::string>("Flux Divergence Name" ,"flux_divergence");
  sliding_velocity_name       = params->sublist("Variables Names").get<std::string>("Sliding Velocity Name" ,"sliding_velocity");

  // Determine what Rigid Body Modes to compute and pass to the preconditioner
  std::string rmb_list_name = "LandIce Rigid Body Modes For Preconditioner";
  if(params->isSublist(rmb_list_name)) {
    auto rbm_list = params_->sublist("LandIce Rigid Body Modes For Preconditioner");
    computeConstantModes = params->sublist(rmb_list_name).get<bool>("Compute Constant Modes");
    computeRotationModes = params->sublist(rmb_list_name).get<bool>("Compute Rotation Modes");
  } else {
    computeConstantModes = true;
    computeRotationModes = false;
  }
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
  if(cellType->getKey() == shards::Wedge<6>::key) {
    std::cout <<  "\nUsing tensor cubature\n" <<std::endl;
    std::vector<int> degree(2); degree[0] = meshSpecs[0]->cubatureDegree;  degree[1] = std::max(meshSpecs[0]->cubatureDegree, this->depthIntegratedModel ? 6 : 4); 
    cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, degree);
  } else {
    cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);
  }

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellType->getSideCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDimFO));

  unsigned int sideDim = numDim-1;
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

      unsigned int numSideVertices = sideType[ssName]->getNodeCount();
      unsigned int numSideNodes    = sideBasis[ssName]->getCardinality();
      unsigned int numSideQPs      = sideCubature[ssName]->getNumPoints();

      dl->side_layouts[ssName] = rcp(new Albany::Layouts(numSideVertices,numSideNodes,
                                                         numSideQPs,sideDim,numDim,numCellSides,vecDimFO,ssName));
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

    dl->side_layouts[surfaceSideName] = rcp(new Albany::Layouts(numSurfaceSideVertices,numSurfaceSideNodes,
                                                                numSurfaceSideQPs,sideDim,numDim,numCellSides,vecDimFO,surfaceSideName));
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

    dl->side_layouts[basalSideName] = rcp(new Albany::Layouts(numbasalSideVertices,numbasalSideNodes,
                                                              numbasalSideQPs,sideDim,numDim,numCellSides,vecDimFO,basalSideName));
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
         << "  Vertices   = " << it.second->vertices_vector->extent(1) << "\n"
         << "  Nodes      = " << it.second->node_scalar->extent(1) << "\n"
         << "  QuadPts    = " << it.second->qp_scalar->extent(1) << "\n";
  }
#endif

  // Parse the input/output fields properties
  // We do this BEFORE the evaluators request/construction,
  // so we have all the info about the input fields.
  parseInputFields ();

  // Set the scalar type of all the fields, plus, if needed, their rank or whether they are computed.
  setFieldsProperties ();

  // Prepare the requests of interpolation/utility evaluators
  setupEvaluatorRequests ();

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);
  buildFields(*fm[0]);

  // Build a dirichlet fm if nodesets are present
  if (meshSpecs[0]->nsNames.size() >0) {
    constructDirichletEvaluators(*meshSpecs[0]);
  }

  // Check if have Neumann sublist; throw error if attempting to specify
  // Neumann BCs, but there are no sidesets in the input mesh
  bool isNeumannPL = params->isSublist("Neumann BCs");
  if (isNeumannPL && !(meshSpecs[0]->ssNames.size() > 0)) {
    ALBANY_ASSERT(false, "You are attempting to set Neumann BCs on a mesh with no sidesets!");
  }

  // Build a neumann fm if sidesets are present
  if(meshSpecs[0]->ssNames.size() > 0) {
     constructNeumannEvaluators(meshSpecs[0]);
  }
}

void
StokesFOBase::buildStokesFOBaseFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  // Allocate memory for unmanaged fields
  fieldUtils->allocateComputeBasisFunctionsFields();
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
  validPL->sublist("Variables Names",false,"Sublist where we can specify a user-defined name for variables.");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("LandIce Field Norm", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("LandIce Noise", false, "");
  validPL->set<bool>("Use Time Parameter", false, "Solely to use Solver Method = Continuation");
  validPL->set<bool>("Print Stress Tensor", false, "Whether to save stress tensor in the mesh");
  validPL->sublist("LandIce Rigid Body Modes For Preconditioner", false, "");
  validPL->set<bool>("Depth Integrated Model", false, "");
  validPL->set<bool>("Depth Integrated Test Functions", false, "");
  return validPL;
}

void StokesFOBase::parseInputFields ()
{
  std::string stateName, fieldName, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  if (this->params->isSublist("Parameters")) {
    const Teuchos::ParameterList& parameterParams = this->params->sublist("Parameters");
    int total_num_param_vecs, num_param_vecs, num_dist_param_vecs;
    Albany::getParameterSizes(parameterParams, total_num_param_vecs, num_param_vecs, num_dist_param_vecs);

    for (unsigned int p_index=0; p_index< (unsigned int) num_dist_param_vecs; ++p_index) {
      std::string parameter_sublist_name = util::strint("Parameter", p_index+num_param_vecs);
      Teuchos::ParameterList param_list = parameterParams.sublist(parameter_sublist_name);
      param_name = param_list.get<std::string>("Name");
      dist_params_name_to_mesh_part[param_name] = param_list.get<std::string>("Mesh Part","");
      is_extruded_param[param_name] = param_list.get<bool>("Extruded",false);
      int extruded_param_level = param_list.get<int>("Extruded Param Level",0);
      extruded_params_levels.insert(std::make_pair(param_name, extruded_param_level));
      save_sensitivities[param_name]=param_list.get<bool>("Save Sensitivity",false);
      is_dist_param[param_name] = true;
      is_input_field[param_name] = true;
      is_dist[param_name] = true;
      is_dist[param_name+"_upperbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_upperbound"] = dist_params_name_to_mesh_part[param_name];
      is_dist[param_name+"_lowerbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_lowerbound"] = dist_params_name_to_mesh_part[param_name];

      // set the scalar type already to ParamScalar
      field_scalar_type[param_name] = FST::ParamScalar;
    }
  }

  //Dirichlet fields need to be distributed but they are not necessarily parameters.
  if (this->params->isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList dirichlet_list = this->params->sublist("Dirichlet BCs");
    for(auto it = dirichlet_list.begin(); it !=dirichlet_list.end(); ++it) {
      std::string pname = dirichlet_list.name(it);
      if(dirichlet_list.isParameter(pname) && dirichlet_list.isType<std::string>(pname)){ //need to check, because pname could be the name sublist
        is_dist[dirichlet_list.get<std::string>(pname)]=true;
        dist_params_name_to_mesh_part[dirichlet_list.get<std::string>(pname)]="";
      }
    }
  }

  // Volume mesh requirements
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  FRT rank;
  FL loc;

  for (int ifield=0; ifield<num_fields; ++ifield) {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(util::strint("Field", ifield));

    // Get current state specs
    fieldName = thisFieldList.get<std::string>("Field Name");
    stateName = thisFieldList.get<std::string>("State Name", fieldName);
    fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    if (fieldUsage == "Unused") {
      continue;
    }

    fieldType  = thisFieldList.get<std::string>("Field Type");

    is_dist_param.insert(std::pair<std::string,bool>(stateName, false));  //gets inserted only if not there.
    is_dist.insert(std::pair<std::string,bool>(stateName, false));        //gets inserted only if not there.

    meshPart = is_dist[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

    if(fieldType.find("Node")!=std::string::npos) {
      loc = FL::Node;
    } else if (fieldType.find("Elem")!=std::string::npos) {
      loc = FL::Cell;
    } else if (fieldType.find("QuadPoint")!=std::string::npos) {
      loc = FL::QuadPoint;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error! Failed to deduce location for field '" + fieldName + "' from type '" + fieldType + "'.\n");
    }

    if(fieldType.find("Scalar")!=std::string::npos) {
      rank = FRT::Scalar;
    } else if (fieldType.find("Vector")!=std::string::npos) {
      rank = FRT::Vector;
    } else if (fieldType.find("Gradient")!=std::string::npos) {
      rank = FRT::Gradient;
    } else if (fieldType.find("Tensor")!=std::string::npos) {
      rank = FRT::Tensor;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error! Failed to deduce rank type for field '" + fieldName + "' from type '" + fieldType + "'.\n");
    }

    // Do we need to load/gather the state/parameter?
    if (is_dist_param[stateName] || fieldUsage == "Input" || fieldUsage == "Input-Output") {
      // A parameter to gather or a field to load
      is_input_field[stateName] = true;
      input_field_loc[stateName] = loc;
    }

    // Set rank, location and scalar type.
    // Note: output fields *may* have ScalarType = Scalar. This may seem like a problem,
    //       but it's not, since the Save(SideSet)StateField evaluators are only created
    //       for the Residual.
    field_rank[stateName] = rank;
    field_scalar_type[stateName] |= FST::Real;

    // Request QP interpolation for all input nodal fields
    if (loc==FL::Node) {
      build_interp_ev[stateName][IReq::QP_VAL] = true;
    }
    if (loc==FL::Cell && !is_input_field[stateName]) {
      build_interp_ev[stateName][IReq::CELL_VAL] = true;
    }
  }

  // Side set requirements
  Teuchos::Array<std::string> ss_names;
  if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
    ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  }
  for (unsigned int i=0; i<ss_names.size(); ++i) {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
    num_fields = info.get<int>("Number Of Fields",0);

    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (unsigned int ifield=0; ifield< (unsigned int) num_fields; ++ifield) {
      Teuchos::ParameterList& thisFieldList =  info.sublist(util::strint("Field", ifield));

      // Get current state specs
      fieldName = thisFieldList.get<std::string>("Field Name");
      stateName = thisFieldList.get<std::string>("State Name", fieldName);
      fieldName = fieldName + "_" + ss_name;
      fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

      if (fieldUsage == "Unused") {
        continue;
      }

      meshPart = ""; // Distributed parameters are defined either on the whole volume mesh or on a whole side mesh. Either way, here we want "" as part (the whole mesh).

      fieldType  = thisFieldList.get<std::string>("Field Type");

      if(fieldType.find("Node")!=std::string::npos) {
        loc = FL::Node;
      } else if (fieldType.find("Elem")!=std::string::npos) {
        loc = FL::Cell;
      } else if (fieldType.find("QuadPoint")!=std::string::npos) {
        loc = FL::QuadPoint;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Error! Failed to deduce location for field '" + fieldName + "' from type '" + fieldType + "'.\n");
      }

      if(fieldType.find("Scalar")!=std::string::npos) {
        rank = FRT::Scalar;
      } else if (fieldType.find("Vector")!=std::string::npos) {
        rank = FRT::Vector;
      } else if (fieldType.find("Gradient")!=std::string::npos) {
        rank = FRT::Gradient;
      } else if (fieldType.find("Tensor")!=std::string::npos) {
        rank = FRT::Tensor;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Error! Failed to deduce rank type for field '" + fieldName + "' from type '" + fieldType + "'.\n");
      }

      if (!is_dist_param[stateName] && (fieldUsage == "Input" || fieldUsage == "Input-Output")) {
        // A parameter to gather or a field to load
        is_ss_input_field[ss_name][stateName] = true;
        ss_input_field_loc[ss_name][stateName] = loc;
      }

      field_rank[stateName] = rank;
      field_scalar_type[stateName] |= FST::Real;

      // Request QP interpolation for all input nodal fields
      if (loc==FL::Node) {
        ss_build_interp_ev[ss_name][stateName][IReq::QP_VAL] = true;
        ss_build_interp_ev[ss_name][stateName][IReq::GRAD_QP_VAL] = rank != FRT::Gradient;
      } else if (!is_ss_input_field[ss_name][stateName]) {
        ss_build_interp_ev[ss_name][stateName][IReq::CELL_VAL] = true;
      }
    }
  }
}

void StokesFOBase::setFieldsProperties ()
{

  // All dofs have scalar type Scalar. Note: we can't set field props for all dofs, since we don't know the rank
  setSingleFieldProperties(velocity_name, FRT::Vector, FST::Scalar);

  // Set rank of known fields. The scalar type will default to RealType.
  // If that's not the case for derived problems, they will update the st.
  setSingleFieldProperties(ice_thickness_name,                FRT::Scalar, FST::MeshScalar);
  setSingleFieldProperties(surface_height_name,               FRT::Scalar, FST::MeshScalar);
  setSingleFieldProperties(vertically_averaged_velocity_name, FRT::Vector, FST::Scalar);
  setSingleFieldProperties(corrected_temperature_name,        FRT::Scalar);
  setSingleFieldProperties(bed_topography_name,               FRT::Scalar, FST::MeshScalar);
  setSingleFieldProperties(body_force_name,                   FRT::Vector);
  setSingleFieldProperties(flow_factor_name,                  FRT::Scalar);
  setSingleFieldProperties(flux_divergence_name,              FRT::Scalar, FST::Scalar);

  setSingleFieldProperties(Albany::coord_vec_name,            FRT::Vector, FST::MeshScalar);
}

void StokesFOBase::setupEvaluatorRequests ()
{
  // Volume required interpolations
  build_interp_ev[dof_names[0]][IReq::QP_VAL] = true;
  build_interp_ev[dof_names[0]][IReq::GRAD_QP_VAL] = true;
  build_interp_ev[surface_height_name][IReq::QP_VAL] = true;
  // If not coupled with cism, we may have to compute the surface gradient ourselves
  build_interp_ev[surface_height_name][IReq::GRAD_QP_VAL] = true;
  if (is_input_field[temperature_name]) {
    build_interp_ev[temperature_name][IReq::CELL_VAL] = true;
  }
  if (is_input_field[stiffening_factor_name]) { 
    build_interp_ev[stiffening_factor_name][IReq::QP_VAL] = true;
  }
  if (viscosity_use_corrected_temperature && is_input_field[surface_height_name]) {
    build_interp_ev[surface_height_name][IReq::CELL_VAL] = true;
    build_interp_ev[corrected_temperature_name][IReq::CELL_VAL] = true;
  }

  build_interp_ev[body_force_name][IReq::CELL_VAL] = true;

  bool has_GLF_resp = false;
  bool has_SMB_resp = false;
  // Basal Friction BC requests
  for (auto it : landice_bcs[LandIceBC::BasalFriction]) {
    std::string ssName = it->get<std::string>("Side Set Name");
    // BasalFriction BC needs velocity on the side, and BFs/coords
    // And if we compute grad beta, we may even need the velocity gradient and the effective pressure gradient
    // (which, if effevtive_pressure is a dist param, needs to be projected to the side)

    ss_utils_needed[ssName][UtilityRequest::BFS] = true;
    ss_utils_needed[ssName][UtilityRequest::NORMALS  ] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;  // Only really needed if stereographic map is used.

    ss_build_interp_ev[ssName][dof_names[0]][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[ssName][dof_names[0]][IReq::QP_VAL      ] = true;
    ss_build_interp_ev[ssName][dof_names[0]][IReq::GRAD_QP_VAL ] = true;
    if (is_input_field["effective_pressure"]) {
      ss_build_interp_ev[ssName][effective_pressure_name][IReq::CELL_TO_SIDE] = true;
      ss_build_interp_ev[ssName][effective_pressure_name][IReq::QP_VAL ] = true;
      ss_build_interp_ev[ssName][effective_pressure_name][IReq::GRAD_QP_VAL ] = true;
    } else if (is_ss_input_field[ssName]["effective_pressure"]) {
      ss_build_interp_ev[ssName][effective_pressure_name][IReq::QP_VAL ] = true;
      ss_build_interp_ev[ssName][effective_pressure_name][IReq::GRAD_QP_VAL ] = true;
    }
    ss_build_interp_ev[ssName][flow_factor_name][IReq::CELL_TO_SIDE] = true;
    setSingleFieldProperties(effective_pressure_name, FRT::Scalar);
    setSingleFieldProperties(sliding_velocity_name,   FRT::Scalar, FST::Scalar);

    auto& bfc = it->sublist("Basal Friction Coefficient");
    const auto type = util::upper_case(bfc.get<std::string>("Type"));
    if (type!="CONSTANT") {
      // For beta type not constant, we might have some auxiliary input fields to interpolate
      if (type == "FIELD") {
        // beta is a given field.
        auto fname = bfc.get<std::string>("Beta Field Name");
        basal_friction_name = fname;
        setSingleFieldProperties(fname, FRT::Scalar, FST::ParamScalar);
        ss_build_interp_ev[ssName][fname][IReq::QP_VAL      ] = true;
        ss_build_interp_ev[ssName][fname][IReq::CELL_TO_SIDE] = true;
        ss_build_interp_ev[ssName][fname][IReq::GRAD_QP_VAL ] = true;
      } else {
        // We have a sliding law, which requires a mu (possibly a field),
        // and possibly a bed roughness
        const auto mu_type = util::upper_case(bfc.get<std::string>("Mu Type"));
        if (mu_type!="CONSTANT") {
          auto mu_field_name = bfc.get<std::string>("Mu Field Name");
          setSingleFieldProperties(mu_field_name, FRT::Scalar, FST::ParamScalar);
          ss_build_interp_ev[ssName][mu_field_name][IReq::QP_VAL      ] = true;
          ss_build_interp_ev[ssName][mu_field_name][IReq::CELL_TO_SIDE] = true;
          ss_build_interp_ev[ssName][mu_field_name][IReq::GRAD_QP_VAL ] = true;
          basal_friction_name = mu_field_name;
        }
        if (type=="REGULARIZED COULOMB") {
          // For Coulomb we *may* have another distributed parameter field.
          // We interpolate (and possibly project on ss) only if it is an inputs
          auto fname = "bed_roughness";
          if (is_input_field[fname] || is_ss_input_field[ssName][fname]) {
            ss_build_interp_ev[ssName][fname][IReq::CELL_TO_SIDE] = true;
            ss_build_interp_ev[ssName][fname][IReq::QP_VAL] = true;
          }
          setSingleFieldProperties(fname, FRT::Scalar);
        }
      }
    }

    if(this->params->isSublist("Response Functions")) {
      const Teuchos::ParameterList& resp = this->params->sublist("Response Functions",true);
      unsigned int num_resps = resp.get<int>("Number Of Responses");
      for(unsigned int i=0; i<num_resps; i++) {
        const std::string responseType = resp.sublist(util::strint("Response", i)).isParameter("Type") ?
         resp.sublist(util::strint("Response", i)).get<std::string>("Type") : std::string("Scalar Response");
        if(responseType == "Sum Of Responses") {
          unsigned int num_sub_resps = resp.sublist(util::strint("Response", i)).get<int>("Number Of Responses");
          for(unsigned int j=0; j<num_sub_resps; j++) {
            const auto& response = resp.sublist(util::strint("Response", i)).sublist(util::strint("Response", j));
            if(response.get<std::string>("Name") == "Grounding Line Flux") {
              has_GLF_resp = true;
              continue;
            }
            if(response.get<std::string>("Name") == "Surface Mass Balance Mismatch") {
              has_SMB_resp = true;
              continue;
            }
            if(response.get<std::string>("Name") == "Squared L2 Difference Side Source ST Target RT") {
              has_SMB_resp = (response.get<std::string>("Source Field Name").find("flux_divergence") != std::string::npos);
              continue;
            }
          }
          if (has_GLF_resp && has_SMB_resp)
            break;
        } else {
          if(resp.sublist(util::strint("Response", i)).get<std::string>("Name") == "Grounding Line Flux") {
            has_GLF_resp = true;
            continue;
          }
          if(resp.sublist(util::strint("Response", i)).get<std::string>("Name") == "Surface Mass Balance Mismatch") {
            has_SMB_resp = true;
            continue;
          }
          if(resp.sublist(util::strint("Response", i)).get<std::string>("Name") == "Squared L2 Difference Side Source ST Target RT") {
            has_SMB_resp = (resp.sublist(util::strint("Response", i)).get<std::string>("Source Field Name").find("flux_divergence") != std::string::npos);
            continue;
          }
        }
        if (has_GLF_resp && has_SMB_resp)
          break;
      }
    }

    if (is_dist_param[bed_topography_param_name] || is_dist_param[surface_height_param_name]) {

      ss_build_interp_ev[ssName][bed_topography_param_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][bed_topography_param_name][IReq::CELL_TO_SIDE] = true;

      ss_build_interp_ev[ssName][bed_topography_observed_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][bed_topography_observed_name][IReq::CELL_TO_SIDE] = true;

      ss_build_interp_ev[ssName][surface_height_param_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][surface_height_param_name][IReq::CELL_TO_SIDE] = true;

      ss_build_interp_ev[ssName][surface_height_observed_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][surface_height_observed_name][IReq::CELL_TO_SIDE] = true;
    }

    { //these are used for several basal boundary conditions
      ss_build_interp_ev[ssName][ice_thickness_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][ice_thickness_name][IReq::CELL_TO_SIDE] = true;

      ss_build_interp_ev[ssName][bed_topography_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][bed_topography_name][IReq::CELL_TO_SIDE] = true;
    }
  }

  // Lateral BC requests
  for (auto it : landice_bcs[LandIceBC::Lateral]) {
    std::string ssName = it->get<std::string>("Side Set Name");

    // Lateral bc needs thickness ...
    ss_build_interp_ev[ssName][ice_thickness_name][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[ssName][ice_thickness_name][IReq::QP_VAL      ] = true;

    // ... possibly surface height ...
    ss_build_interp_ev[ssName][surface_height_name][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[ssName][surface_height_name][IReq::QP_VAL      ] = true;

    // ... and BFs (including normals)
    ss_utils_needed[ssName][UtilityRequest::BFS      ] = true;
    ss_utils_needed[ssName][UtilityRequest::NORMALS  ] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;
  }

  // Surface diagnostics
  if (!isInvalid(surfaceSideName)) {
    // Surface velocity diagnostic requires dof at qps on surface side
    ss_build_interp_ev[surfaceSideName][dof_names[0]][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[surfaceSideName][dof_names[0]][IReq::QP_VAL      ] = true;

    // ... and observed surface velocity ...
    // NOTE: RMS could be either scalar or vector. The states registration should have figure it out (if the field is listed as input).
    // Note: for input fields, do not specify location, since it should be already set
    if (is_ss_input_field[surfaceSideName]["observed_surface_velocity"]) {
      ss_build_interp_ev[surfaceSideName]["observed_surface_velocity"][IReq::QP_VAL] = true;
    }
    if (is_ss_input_field[surfaceSideName]["observed_surface_velocity_RMS"]) {
      ss_build_interp_ev[surfaceSideName]["observed_surface_velocity_RMS"][IReq::QP_VAL] = true;
    }

    // ... and BFs
    ss_utils_needed[surfaceSideName][UtilityRequest::BFS] = true;

    if (!isInvalid(basalSideName) && is_input_field[stiffening_factor_name]) {
      // Surface velocity diagnostics *may* add a basal side regularization
      ss_build_interp_ev[basalSideName][stiffening_factor_name][IReq::CELL_TO_SIDE] = true;
      ss_build_interp_ev[basalSideName][stiffening_factor_name][IReq::QP_VAL      ] = true;
      ss_build_interp_ev[basalSideName][stiffening_factor_name][IReq::GRAD_QP_VAL ] = true;

      ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;
    }
  }
  
  // // Needed for computing flux divergence when has_SMB_resp == flase
  // if (!isInvalid(basalSideName)) {
  //   ss_build_interp_ev[basalSideName][flux_divergence_name][IReq::CELL_VAL] = true;
  //   ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;
  //   ss_build_interp_ev[basalSideName][vertically_averaged_velocity_name][IReq::QP_VAL] = true;
  // }

  // SMB-related diagnostics
  if (!isInvalid(basalSideName) && has_SMB_resp) {
    // Needs BFs
    ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;

    // SMB evaluators may need velocity, averaged velocity, and thickness
    ss_build_interp_ev[basalSideName][dof_names[0]][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[basalSideName][dof_names[0]][IReq::QP_VAL      ] = true;
    ss_build_interp_ev[basalSideName][vertically_averaged_velocity_name][IReq::QP_VAL] = true;
    ss_build_interp_ev[basalSideName][ice_thickness_name][IReq::QP_VAL      ] = true;
    ss_build_interp_ev[basalSideName][ice_thickness_name][IReq::GRAD_QP_VAL ] = true;
    ss_build_interp_ev[basalSideName][ice_thickness_name][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[basalSideName][flux_divergence_name][IReq::CELL_VAL] = true;
    if (is_ss_input_field[basalSideName]["apparent_mass_balance"]) {
      ss_build_interp_ev[basalSideName]["apparent_mass_balance"][IReq::QP_VAL] = true;
    }
    if (is_ss_input_field[basalSideName]["apparent_mass_balance_RMS"]) {
      ss_build_interp_ev[basalSideName]["apparent_mass_balance_RMS"][IReq::QP_VAL] = true;
    }
    if (is_ss_input_field[basalSideName]["observed_ice_thickness"]) {
      ss_build_interp_ev[basalSideName]["observed_ice_thickness"][IReq::QP_VAL] = true;
    }
    if (is_ss_input_field[basalSideName]["observed_ice_thickness_RMS"]) {
      ss_build_interp_ev[basalSideName]["observed_ice_thickness_RMS"][IReq::QP_VAL] = true;
    }
  }
}

void StokesFOBase::setSingleFieldProperties (const std::string& fname,
                                             const FRT rank,
                                             const FST st)
{
  TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)!=field_rank.end() &&
                              field_rank[fname]!=rank, std::logic_error,
      "Error! Attempt to mark field '" + fname + " with rank " + e2str(rank) + "\n"
      "       but it was previously marked with rank " + e2str(field_rank[fname]) + ".\n");

  field_rank[fname] = rank;
  // Use logic or, so that if the props of a field are set multiple times
  // (e.g., in base and derived classes), we keep the "strongest" scalar type.
  field_scalar_type[fname] |= st;
}

FieldScalarType StokesFOBase::get_scalar_type (const std::string& fname) {
  // Note: if not set, it defaults to simple real type, the "weakest" scalar type
  FST st = field_scalar_type[fname];
  const auto& deps = field_deps[fname];
  for (const auto& dep : deps) {
    st |= get_scalar_type(dep);
  }

  return st;
}

Albany::FieldRankType StokesFOBase::get_field_rank (const std::string& fname) {
  // Note: if not set, it defaults to Scalar, but printing a warning
  auto it = field_rank.find(fname);

  if (it==field_rank.end()) {
#ifndef NDEBUG
    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    int commRank = Teuchos::GlobalMPISession::getRank();
    int commSize = Teuchos::GlobalMPISession::getNProc();
    out->setProcRankAndSize(commRank, commSize);
    out->setOutputToRootOnly(0);
    *out << " ** Rank of field '" << fname << "' not set. Defaulting to 'Scalar'.\n";
#endif
    return FRT::Scalar;
  }
  return it->second;
}

void StokesFOBase::add_dep (const std::string& fname, const std::string& dep_name) {
  field_deps[fname].insert(dep_name);
}

} // namespace LandIce
