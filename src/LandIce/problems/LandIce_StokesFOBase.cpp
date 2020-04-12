#include "LandIce_StokesFOBase.hpp"
#include "Teuchos_CompilerCodeTweakMacros.hpp"

#include <string.hpp>               // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)
#include <Albany_ProblemUtils.hpp>  // For 'getIntrepidwBasis'

namespace LandIce {

StokesFOBase::
StokesFOBase (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                        const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                        const Teuchos::RCP<ParamLib>& paramLib_,
                        const int numDim_)
 : Albany::AbstractProblem(params_, paramLib_, numDim_)
 , params(params_) 
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

  Teuchos::ParameterList& physics_list = params->sublist("LandIce Physical Parameters");

  if (physics_list.isParameter("Clausius-Clapeyron Coefficient") &&
      physics_list.get<double>("Clausius-Clapeyron Coefficient")!=0.0) {
    viscosity_use_corrected_temperature = true;
  } else {
    viscosity_use_corrected_temperature = false;
  }
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

  dof_names[0] = "Velocity";
  resid_names[0] = dof_names[0] + " Residual";
  scatter_names[0] = "Scatter " + resid_names[0];

  dof_offsets.resize(1);
  dof_offsets[0] = 0;
  vecDimFO = std::min((int)neq,(int)2);

  // Names of some common fields. User can set them in the problem section, in case they are
  // loaded from mesh, where they are saved with a different name
  body_force_name             = params->sublist("Variables Names").get<std::string>("Body Force Name","body_force");
  surface_height_name         = params->sublist("Variables Names").get<std::string>("Surface Height Name","surface_height");
  ice_thickness_name          = params->sublist("Variables Names").get<std::string>("Ice Thickness Name" ,"ice_thickness");
  temperature_name            = params->sublist("Variables Names").get<std::string>("Temperature Name"   ,"temperature");
  corrected_temperature_name  = "corrected_" + temperature_name;
  bed_topography_name         = params->sublist("Variables Names").get<std::string>("Bed Topography Name"    ,"bed_topography");
  flow_factor_name            = params->sublist("Variables Names").get<std::string>("Flow Factor Name"       ,"flow_factor");
  stiffening_factor_name      = params->sublist("Variables Names").get<std::string>("Stiffening Factor Name" ,"stiffening_factor");
  effective_pressure_name     = params->sublist("Variables Names").get<std::string>("Effective Pressure Name","effective_pressure");
  vertically_averaged_velocity_name = params->sublist("Variables Names").get<std::string>("Vertically Averaged Velocity Name","vertically_averaged_velocity");
  flux_divergence_name        = params->sublist("Variables Names").get<std::string>("Flux Divergence Name" ,"flux_divergence");

  // Mark the velocity as computed
  is_computed_field[dof_names[0]] = true;

  // By default, we are not coupled to any other physics
  temperature_coupled = false;
  hydrology_coupled = false;
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

  return validPL;
}

void StokesFOBase::parseInputFields ()
{
  std::string stateName, fieldName, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  if (this->params->isSublist("Distributed Parameters")) {
    Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
    Teuchos::ParameterList* param_list;
    int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
    for (int p_index=0; p_index< numParams; ++p_index) {
      std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
      if (dist_params_list.isSublist(parameter_sublist_name)) {
        // The better way to specify dist params: with sublists
        param_list = &dist_params_list.sublist(parameter_sublist_name);
        param_name = param_list->get<std::string>("Name");
        dist_params_name_to_mesh_part[param_name] = param_list->get<std::string>("Mesh Part","");
        is_extruded_param[param_name] = param_list->get<bool>("Extruded",false);
        int extruded_param_level = param_list->get<int>("Extruded Param Level",0);
        extruded_params_levels.insert(std::make_pair(param_name, extruded_param_level));
        save_sensitivities[param_name]=param_list->get<bool>("Save Sensitivity",false);
      } else {
        // Legacy way to specify dist params: with parameter entries. Note: no mesh part can be specified.
        param_name = dist_params_list.get<std::string>(Albany::strint("Parameter", p_index));
        dist_params_name_to_mesh_part[param_name] = "";
      }
      is_dist_param[param_name] = true;
      is_input_field[param_name] = true;
      is_dist[param_name] = true;
      is_dist[param_name+"_upperbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_upperbound"] = dist_params_name_to_mesh_part[param_name];
      is_dist[param_name+"_lowerbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_lowerbound"] = dist_params_name_to_mesh_part[param_name];

      // set hte scalar type already to ParamScalar
      field_scalar_type[param_name] = FieldScalarType::ParamScalar;
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
  bool nodal_state, scalar_state;
  for (int ifield=0; ifield<num_fields; ++ifield) {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

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

    if(fieldType == "Elem Scalar") {
      nodal_state = false;
      scalar_state = true;
    } else if(fieldType == "Node Scalar") {
      nodal_state = true;
      scalar_state = true;
    } else if(fieldType == "Elem Vector") {
      nodal_state = false;
      scalar_state = false;
    } else if(fieldType == "Node Vector") {
      nodal_state = true;
      scalar_state = false;
    }

    // Do we need to load/gather the state/parameter?
    if (is_dist_param[stateName] || fieldUsage == "Input" || fieldUsage == "Input-Output") {
      // A parameter to gather or a field to load
      is_input_field[stateName] = true;
    }

    // Set rank, location and scalar type.
    // Note: output fields *may* have ScalarType = Scalar. This may seem like a problem,
    //       but it's not, since the Save(SideSet)StateField evaluators are only created
    //       for the Residual.
    field_rank[stateName] = scalar_state ? 0 : 1;
    field_location[stateName] = nodal_state ? FieldLocation::Node : FieldLocation::Cell;
    field_scalar_type[stateName] |= FieldScalarType::Real;
  }

  // Side set requirements
  Teuchos::Array<std::string> ss_names;
  if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
    ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  } 
  for (int i=0; i<ss_names.size(); ++i) {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
    num_fields = info.get<int>("Number Of Fields",0);

    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (int ifield=0; ifield<num_fields; ++ifield) {
      Teuchos::ParameterList& thisFieldList =  info.sublist(Albany::strint("Field", ifield));

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

      if(fieldType == "Elem Scalar") {
        nodal_state = false;
        scalar_state = true;
      } else if(fieldType == "Node Scalar") {
        nodal_state = true;
        scalar_state = true;
      } else if(fieldType == "Elem Vector") {
        nodal_state = false;
        scalar_state = false;
      } else if(fieldType == "Node Vector") {
        nodal_state = true;
        scalar_state = false;
      } else if(fieldType == "Elem Layered Scalar") {
        nodal_state = false;
        scalar_state = true;
      } else if(fieldType == "Node Layered Scalar") {
        nodal_state = true;
        scalar_state = true;
      } else if(fieldType == "Elem Layered Vector") {
        nodal_state = false;
        scalar_state = false;
      } else if(fieldType == "Node Layered Vector") {
        nodal_state = true;
        scalar_state = false;
      }

      if (!is_dist_param[stateName] && (fieldUsage == "Input" || fieldUsage == "Input-Output")) {
        // A parameter to gather or a field to load
        is_ss_input_field[ss_name][stateName] = true;
      }

      field_rank[stateName] = scalar_state ? 0 : 1;
      field_location[stateName] = nodal_state ? FieldLocation::Node : FieldLocation::Cell;
      field_scalar_type[stateName] |= FieldScalarType::Real;
    }
  }
}

void StokesFOBase::setFieldsProperties ()
{
  // All dofs are computed
  for (auto it : dof_names) {
    is_computed_field[it] = true;
  }

  // All dofs have scalar type Scalar. Note: we can't set field props for all dofs, since we don't know the rank
  setSingleFieldProperties(dof_names[0], 1, FieldScalarType::Scalar, FieldLocation::Node);

  // Set properties of known fields. If things are different in derived classes, then adjust.
  setSingleFieldProperties(ice_thickness_name,  0, FieldScalarType::MeshScalar, FieldLocation::Node);
  setSingleFieldProperties(surface_height_name, 0, FieldScalarType::MeshScalar, FieldLocation::Node);
  setSingleFieldProperties(vertically_averaged_velocity_name, 1, FieldScalarType::Scalar, FieldLocation::Node);
  // Note: corr temp is built from temp and surf height. Combine their scalar types.
  //       If derived class changes the type of temp or surf height, need to adjust this too.
  setSingleFieldProperties(corrected_temperature_name, 0, field_scalar_type[temperature_name] | field_scalar_type[surface_height_name], FieldLocation::Cell);
  setSingleFieldProperties(bed_topography_name, 0, FieldScalarType::MeshScalar, FieldLocation::Node);
  setSingleFieldProperties(body_force_name, 1, field_scalar_type[surface_height_name], FieldLocation::QuadPoint);

  // If the flow rate is given from file, we could just use RealType, but then we would need
  // to template ViscosityFO on 3 scalar types. For simplicity, we set it to be the same
  // of the temperature in ViscosityFO.
  if (viscosity_use_corrected_temperature) {
    setSingleFieldProperties(flow_factor_name, 0, FieldScalarType::MeshScalar | field_scalar_type[corrected_temperature_name], FieldLocation::Cell);
  } else {
    setSingleFieldProperties(flow_factor_name, 0, FieldScalarType::MeshScalar | field_scalar_type[temperature_name], FieldLocation::Cell);
  }
}

void StokesFOBase::setupEvaluatorRequests ()
{
  // Volume required interpolations
  build_interp_ev[dof_names[0]][InterpolationRequest::QP_VAL] = true; 
  build_interp_ev[dof_names[0]][InterpolationRequest::GRAD_QP_VAL] = true; 
  build_interp_ev[surface_height_name][InterpolationRequest::QP_VAL] = true; 
  // If not coupled with cism, we may have to compute the surface gradient ourselves
  build_interp_ev[surface_height_name][InterpolationRequest::GRAD_QP_VAL] = true; 
  if (is_input_field[temperature_name]) {
    build_interp_ev[temperature_name][InterpolationRequest::CELL_VAL] = true;
  }
  if (is_input_field[stiffening_factor_name]) {
    build_interp_ev[stiffening_factor_name][InterpolationRequest::QP_VAL] = true; 
  }
  if (viscosity_use_corrected_temperature && is_input_field[surface_height_name]) {
    build_interp_ev[surface_height_name][InterpolationRequest::CELL_VAL] = true;
    build_interp_ev[corrected_temperature_name][InterpolationRequest::CELL_VAL] = true;
  }

  build_interp_ev[body_force_name][InterpolationRequest::CELL_VAL] = true;

  // Basal Friction BC requests
  for (auto it : landice_bcs[LandIceBC::BasalFriction]) {
    std::string ssName = it->get<std::string>("Side Set Name");
    // BasalFriction BC needs velocity on the side, and BFs/coords
    // And if we compute grad beta, we may even need the velocity gradient and the effective pressure gradient
    // (which, if effevtive_pressure is a dist param, needs to be projected to the side)

    ss_utils_needed[ssName][UtilityRequest::BFS] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;  // Only really needed if stereographic map is used.

    ss_build_interp_ev[ssName][dof_names[0]][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[ssName][dof_names[0]][InterpolationRequest::QP_VAL      ] = true; 
    ss_build_interp_ev[ssName][dof_names[0]][InterpolationRequest::GRAD_QP_VAL ] = true; 
    if (is_input_field["effective_pressure"] || is_ss_input_field[ssName]["effective_pressure"])
      ss_build_interp_ev[ssName][effective_pressure_name][InterpolationRequest::QP_VAL ] = true;
    ss_build_interp_ev[ssName][effective_pressure_name][InterpolationRequest::GRAD_QP_VAL ] = true; 
    ss_build_interp_ev[ssName][effective_pressure_name][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[ssName][flow_factor_name][InterpolationRequest::CELL_TO_SIDE] = true;

    // These two are needed for coulomb friction
    ss_build_interp_ev[ssName]["mu_coulomb"][InterpolationRequest::QP_VAL] = true;
    ss_build_interp_ev[ssName]["mu_power_law"][InterpolationRequest::QP_VAL] = true;
    ss_build_interp_ev[ssName]["bed_roughness"][InterpolationRequest::QP_VAL] = true;

    if (is_dist_param["mu_coulomb"]) {
      ss_build_interp_ev[ssName]["mu_coulomb"][InterpolationRequest::CELL_TO_SIDE] = true;
    }
    if (is_dist_param["mu_power_law"]) {
      ss_build_interp_ev[ssName]["mu_power_law"][InterpolationRequest::CELL_TO_SIDE] = true;
    }
    if (is_dist_param["bed_roughness"]) {
      ss_build_interp_ev[ssName]["bed_roughness"][InterpolationRequest::CELL_TO_SIDE] = true;
    }

    // For "Given Field" and "Exponent of Given Field" we also need to interpolate the given field at the quadrature points
    auto& bfc = it->sublist("Basal Friction Coefficient");
    const auto type = util::upper_case(bfc.get<std::string>("Type"));
    if (type=="GIVEN FIELD" || type=="EXPONENT OF GIVEN FIELD") {
      ss_build_interp_ev[ssName][bfc.get<std::string>("Given Field Variable Name")][InterpolationRequest::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][bfc.get<std::string>("Given Field Variable Name")][InterpolationRequest::CELL_TO_SIDE] = true;
    }

    bool has_GLF_resp = false;
    if(this->params->isSublist("Response Functions")) {
      auto resp = this->params->sublist("Response Functions",true);
      int num_resps = resp.get("Number", 0);
      for(int i=0; i<num_resps; i++) {
        if(resp.get<std::string>(Albany::strint("Response", i)) == "Grounding Line Flux") {
          has_GLF_resp = true;
          break;
        }
      }
    }

    // If zero on floating, we also need bed topography and thickness
    if (bfc.get<bool>("Zero Beta On Floating Ice", false) || has_GLF_resp) {
      ss_build_interp_ev[ssName][bed_topography_name][InterpolationRequest::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][bed_topography_name][InterpolationRequest::CELL_TO_SIDE] = true;

      ss_build_interp_ev[ssName][ice_thickness_name][InterpolationRequest::QP_VAL      ] = true;
      ss_build_interp_ev[ssName][ice_thickness_name][InterpolationRequest::CELL_TO_SIDE] = true;
    }
  }

  // Lateral BC requests
  for (auto it : landice_bcs[LandIceBC::Lateral]) {
    std::string ssName = it->get<std::string>("Side Set Name");

    // Lateral bc needs thickness ...
    ss_build_interp_ev[ssName][ice_thickness_name][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[ssName][ice_thickness_name][InterpolationRequest::QP_VAL      ] = true; 

    // ... possibly surface height ...
    ss_build_interp_ev[ssName][surface_height_name][InterpolationRequest::CELL_TO_SIDE] = true;
    ss_build_interp_ev[ssName][surface_height_name][InterpolationRequest::QP_VAL      ] = true; 

    // ... and BFs (including normals)
    ss_utils_needed[ssName][UtilityRequest::BFS      ] = true;
    ss_utils_needed[ssName][UtilityRequest::NORMALS  ] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;
  }

  // Surface diagnostics
  if (!isInvalid(surfaceSideName)) {
    // Surface velocity diagnostic requires dof at qps on surface side
    ss_build_interp_ev[surfaceSideName][dof_names[0]][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[surfaceSideName][dof_names[0]][InterpolationRequest::QP_VAL      ] = true; 

    // ... and observed surface velocity ...
    // NOTE: RMS could be either scalar or vector. The states registration should have figure it out (if the field is listed as input).
    // Note: for input fields, do not specify location, since it should be already set
    if (is_ss_input_field[surfaceSideName]["observed_surface_velocity"]) {
      ss_build_interp_ev[surfaceSideName]["observed_surface_velocity"][InterpolationRequest::QP_VAL] = true;
    }
    if (is_ss_input_field[surfaceSideName]["observed_surface_velocity_RMS"]) {
      ss_build_interp_ev[surfaceSideName]["observed_surface_velocity_RMS"][InterpolationRequest::QP_VAL] = true; 
    }

    // ... and BFs
    ss_utils_needed[surfaceSideName][UtilityRequest::BFS] = true;

    if (!isInvalid(basalSideName) && is_input_field[stiffening_factor_name]) {
      // Surface velocity diagnostics *may* add a basal side regularization
      ss_build_interp_ev[basalSideName][stiffening_factor_name][InterpolationRequest::CELL_TO_SIDE] = true; 
      ss_build_interp_ev[basalSideName][stiffening_factor_name][InterpolationRequest::QP_VAL      ] = true; 
      ss_build_interp_ev[basalSideName][stiffening_factor_name][InterpolationRequest::GRAD_QP_VAL ] = true; 

      ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;
    }
  }

  // SMB-related diagnostics
  if (!isInvalid(basalSideName)) {
    // Needs BFs
    ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;

    // SMB evaluators may need velocity, averaged velocity, and thickness
    ss_build_interp_ev[basalSideName][dof_names[0]][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[basalSideName][dof_names[0]][InterpolationRequest::QP_VAL      ] = true; 
    ss_build_interp_ev[basalSideName][vertically_averaged_velocity_name][InterpolationRequest::QP_VAL] = true; 
    ss_build_interp_ev[basalSideName][ice_thickness_name][InterpolationRequest::QP_VAL      ] = true; 
    ss_build_interp_ev[basalSideName][ice_thickness_name][InterpolationRequest::GRAD_QP_VAL ] = true; 
    ss_build_interp_ev[basalSideName][ice_thickness_name][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[basalSideName][flux_divergence_name][InterpolationRequest::CELL_VAL] = true;
    if (is_ss_input_field[basalSideName]["apparent_mass_balance"]) {
      ss_build_interp_ev[basalSideName]["apparent_mass_balance"][InterpolationRequest::QP_VAL] = true;
    }
    if (is_ss_input_field[basalSideName]["apparent_mass_balance_RMS"]) {
      ss_build_interp_ev[basalSideName]["apparent_mass_balance_RMS"][InterpolationRequest::QP_VAL] = true;
    }
    if (is_ss_input_field[basalSideName]["observed_ice_thickness"]) {
      ss_build_interp_ev[basalSideName]["observed_ice_thickness"][InterpolationRequest::QP_VAL] = true; 
    }
    if (is_ss_input_field[basalSideName]["observed_ice_thickness_RMS"]) {
      ss_build_interp_ev[basalSideName]["observed_ice_thickness_RMS"][InterpolationRequest::QP_VAL] = true;
    }
  }
}

void StokesFOBase::setSingleFieldProperties (const std::string& fname,
                                             const int rank,
                                             const FieldScalarType st,
                                             const FieldLocation location)
{
  TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)!=field_rank.end() && field_rank[fname]!=rank, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " with rank " + std::to_string(rank) +
                              ", when it was previously marked as of rank " + std::to_string(field_rank[fname]) + ".\n");

  field_rank[fname] = rank;
  field_location[fname] = location;
  field_scalar_type[fname] |= st; // Use logic or, so that if the props of a field are set multiple times (in base and derived), we keep the strongest
}

} // namespace LandIce
