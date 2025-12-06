#include "Albany_ExtrudedMesh.hpp"

#include "Albany_DiscretizationUtils.hpp"

namespace Albany {

ExtrudedMesh::
ExtrudedMesh (const Teuchos::RCP<AbstractMeshStruct>& basal_mesh,
              const Teuchos::RCP<Teuchos::ParameterList>& params,
              const Teuchos::RCP<const Teuchos_Comm>& comm)
 : m_comm (comm)
 , m_params (params)
 , m_basal_mesh (basal_mesh)
{
  // Sanity checks
  TEUCHOS_TEST_FOR_EXCEPTION (basal_mesh.is_null(), std::invalid_argument,
      "[ExtrudedMesh] Error! Invalid basal mesh pointer.\n");
  sideSetMeshStructs["basalside"] = m_basal_mesh;

  const auto basal_mesh_specs = m_basal_mesh->meshSpecs[0];
  const int basalNumDim = basal_mesh_specs->numDim;

  TEUCHOS_TEST_FOR_EXCEPTION (basalNumDim!=2, std::logic_error,
      "[ExtrudedMesh] Error! ExtrudedMesh only available in 3D.\n"
      "  - basal mesh dim: " << basalNumDim << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (m_params.is_null(), std::runtime_error,
      "[ExtrudedMesh] Error! Invalid parameter list pointer.\n");

  // Create layered mesh numbering objects
  constexpr auto COLUMN = LayeredMeshOrdering::COLUMN;
  constexpr auto LAYER  = LayeredMeshOrdering::LAYER;
  const auto num_layers = m_params->get<int>("NumLayers");
  const auto ordering = m_params->get("Columnwise Ordering", false) ? COLUMN : LAYER;
  TEUCHOS_TEST_FOR_EXCEPTION (num_layers<=0, Teuchos::Exceptions::InvalidParameterValue,
      "[ExtrudedMesh] Error! Number of layers must be strictly positive.\n"
      "  - NumLayers: " << num_layers << "\n");

  m_elem_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(num_layers,ordering));
  m_elem_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_layers,COLUMN));
  m_node_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(num_layers+1,ordering));
  m_node_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_layers+1,COLUMN));

  // Create map part->basal_part
  // upper/basal/bot/top parts map to the full basal mesh
  m_part_to_basal_part["basalside"] = m_basal_mesh->meshSpecs[0]->ebName;
  m_part_to_basal_part["upperside"] = m_basal_mesh->meshSpecs[0]->ebName;
  m_part_to_basal_part["bottom"] = "";
  m_part_to_basal_part["top"] = "";

  // Create extruded sideSets/nodeSets/elemBlocks names
  std::vector<std::string> nsNames = {"lateral", "bottom", "top"};
  std::vector<std::string> ssNames = {"lateralside", "basalside", "upperside"};
  std::vector<std::string> lateralParts = {"lateralside"};

  for (const auto& ns : basal_mesh_specs->nsNames) {
    if (ns=="lateral") continue; // extruded_lateral would just be the same as lateral
    nsNames.push_back ("extruded_" + ns);
    nsNames.push_back ("basal_" + ns);

    m_part_to_basal_part["extruded_" + ns] = ns;
    m_part_to_basal_part["basal_" + ns] = ns;
  }
  std::vector<std::string> extruded_ss_names;
  for (const auto& ss : basal_mesh_specs->ssNames) {
    if (ss=="lateralside") continue; // extruded_lateralside would just be the same as lateralside
    auto pname = "extruded_" + ss;
    ssNames.push_back (pname);
    lateralParts.push_back(pname);
    m_part_to_basal_part["extruded_" + ss] = ss;
    extruded_ss_names.push_back(pname);
  }
  std::string ebName = "extruded_" + basal_mesh_specs->ebName;
  std::map<std::string,int> ebNameToIndex =
  {
    { ebName, 0}
  };

  // The whole mesh maps to the full basal mesh
  m_part_to_basal_part[ebName] = basal_mesh_specs->ebName;
  m_part_to_basal_part[""] = "";

  // Determine topology
  auto basal_topo = basal_mesh_specs->ctd;
  auto tria_topo = *shards::getCellTopologyData<shards::Triangle<3> >();
  auto quad_topo = *shards::getCellTopologyData<shards::Quadrilateral<4>>();
  auto wedge_topo = *shards::getCellTopologyData<shards::Wedge<6>>();
  CellTopologyData elem_topo, lat_topo;
  if (basal_topo.name==tria_topo.name) {
    elem_topo = wedge_topo;
    lat_topo  = quad_topo;
  } else if (basal_topo.name==quad_topo.name) {
    elem_topo = wedge_topo;
    lat_topo  = quad_topo;
  } else {
    throw Teuchos::Exceptions::InvalidParameterValue(
      "[ExtrudedMeshStruct] Invalid/unsupported basal mesh element type.\n"
      "  - valid element types: " + std::string(tria_topo.name) + ", " + std::string(quad_topo.name) + "\n"
      "  - basal alement type : " + std::string(basal_topo.name) + "\n");
  }

  // Compute workset size
  int basalWorksetSize = basal_mesh_specs->worksetSize;
  int worksetSizeMax = m_params->get<int>("Workset Size");
  int ebSizeMaxEstimate = basalWorksetSize * num_layers; // This is ebSizeMax when basalWorksetSize is max
  int worksetSize = computeWorksetSize(worksetSizeMax, ebSizeMaxEstimate);

  // Finally, we can create the mesh specs
  this->meshSpecs.resize(1,Teuchos::rcp(
        new MeshSpecsStruct(MeshType::Extruded, elem_topo, basalNumDim+1, nsNames, ssNames,
                            worksetSize, ebName, ebNameToIndex)));

  // Create basalside, uppserside, and lateralside mesh specs
  auto& ss_ms = meshSpecs[0]->sideSetMeshSpecs;

  ss_ms["basalside"] = m_basal_mesh->meshSpecs;

  // Create mesh specs for upper and lateral sides, as well as any extruded side set.
  // We cannot assume there will be a discretization there, so the mesh specs is "empty"
  // (meaning only topology and dim). If a side disc is created (which is AFTER the
  // mesh is created) they will be overwritten.
  auto create_ss_mesh_specs = [&](const std::string& ss_name, const auto& topo)
  {
    auto& ss_ms = meshSpecs[0]->sideSetMeshSpecs[ss_name];
    ss_ms.resize(1);
    ss_ms[0] = Teuchos::rcp( new MeshSpecsStruct() );
    ss_ms[0]->ctd = topo;
    ss_ms[0]->numDim = meshSpecs[0]->numDim-1;
  };
  create_ss_mesh_specs("upperside",basal_topo);
  create_ss_mesh_specs("lateralside",lat_topo);
  for (auto ss_name : extruded_ss_names)
    create_ss_mesh_specs(ss_name,lat_topo);

  // For the upperside, we use the same disc as the basalside.
  sideSetMeshStructs["upperside"] = m_basal_mesh;
}

void ExtrudedMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& sis,
              std::map<std::string, Teuchos::RCP<StateInfoStruct> > side_set_sis)
{
  // Make sure we can dereference sis
  if (sis.is_null()) {
    auto nonnull_sis = Teuchos::rcp(new StateInfoStruct());
    this->setFieldData(comm,nonnull_sis,side_set_sis);
  }

  // Make sure we can dereference the basal sis
  auto basal_sis = side_set_sis["basalside"];
  if (basal_sis.is_null()) {
    basal_sis = Teuchos::rcp(new StateInfoStruct());
  }

  const auto& upper_sis = side_set_sis["upperside"];
  if (not upper_sis.is_null()) {
    for (auto st : *upper_sis) {
      basal_sis->push_back(st);
    }
  }

  auto NDTEN = StateStruct::MeshFieldEntity::NodalDataToElemNode;

  // If we extrude or interpolate a basal state, we will have a name clash.
  // In fact, the ExtrudedMeshFieldAccessor will use the basal mesh field accessor
  // to store the 3d states. This, in turn, is a pb when putting those fields on the
  // mesh, as the underlying mesh database won't like 2 declarations of the same field
  // with different layouts. Hence, we need to disambiguate the states.
  // We use this rules:
  //  - extruded fields: the 2d field keeps the original name, and the 3d one will NOT
  //    be added to the mesh. That is, the 3d state array will be managed by the field accessor.
  //  - interpolated fields: the basal layered field (whose number of layers may not match the
  //    mesh number of layers) will be renamed ${name}_layers_data. We will also register a 3d
  //    state with name ${name}, and layout compatible with the mesh actual number of layers
  const auto& extrude_names = m_params->get<Teuchos::Array<std::string>>("Extrude Basal Fields",{});
  const auto& interpolate_names = m_params->get<Teuchos::Array<std::string>>("Interpolate Basal Layered Fields",{});

  for (const auto& n : extrude_names) {
    TEUCHOS_TEST_FOR_EXCEPTION (basal_sis->find(n,false).is_null(), std::runtime_error,
        "Error! Cannot extrude basal state '" + n + "'. State not found in basal SIS.\n");

    auto st = sis->find(n);
    st->extruded = true;
  }

  for (const auto& n : interpolate_names) {
    TEUCHOS_TEST_FOR_EXCEPTION (basal_sis->find(n,false).is_null(), std::runtime_error,
        "Error! Cannot interpolate basal state '" + n + "'. State not found in basal SIS.\n");

    auto st = sis->find(n);
    st->interpolated = true;
  }

  // Ensure surface_height and thickness are valid states in the basal mesh
  std::string thickness_name = m_params->get<std::string>("Thickness Field Name","thickness");
  std::string surface_height_name = m_params->get<std::string>("Surface Height Field Name","surface_height");
  PHX::DataLayout::size_type basal_wss = m_basal_mesh->meshSpecs[0]->worksetSize;
  PHX::DataLayout::size_type basal_nc  = m_basal_mesh->meshSpecs[0]->ctd.node_count;
  if (basal_sis->find(surface_height_name,false).is_null()) {
    auto st = basal_sis->emplace_back(Teuchos::rcp(new StateStruct(surface_height_name,NDTEN)));
    st->dim = {basal_wss,basal_nc};
  }
  if (basal_sis->find(thickness_name,false).is_null()) {
    auto st = basal_sis->emplace_back(Teuchos::rcp(new StateStruct(thickness_name,NDTEN)));
    st->dim = {basal_wss,basal_nc};
  }

  // Ensure field data is set on basal mesh
  // Since we store upper and basal stuff on same mesh, pass both basal and upper SIS
  if (not side_set_sis["upperside"].is_null()) {
    for (auto st : *side_set_sis["upperside"]) {
      basal_sis->push_back(st);
    }
  }
  m_basal_mesh->setFieldData(comm,basal_sis,{});

  // Now the basal field accessor is definitely valid/inited, so we can create the extruded one
  m_field_accessor = Teuchos::rcp(new ExtrudedMeshFieldAccessor(m_basal_mesh->get_field_accessor(),
                                                                m_elem_layers_data_lid));

  m_field_accessor->addStateStructs(sis);

  m_field_data_set = true;
}

void ExtrudedMesh::
setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  if (not m_basal_mesh->isBulkDataSet()) {
    m_basal_mesh->setBulkData(comm);
  }

  // Complete initialization of layer data structures
  const auto max_basal_node_gid = m_basal_mesh->get_max_node_gid();
  const auto num_basal_nodes    = m_basal_mesh->get_num_local_nodes();
  const auto max_basal_elem_gid = m_basal_mesh->get_max_elem_gid();
  const auto num_basal_elems    = m_basal_mesh->get_num_local_elements();

  m_elem_layers_data_gid->numHorizEntities = max_basal_elem_gid+1;
  m_elem_layers_data_lid->numHorizEntities = num_basal_elems;
  m_node_layers_data_gid->numHorizEntities = max_basal_node_gid+1;
  m_node_layers_data_lid->numHorizEntities = num_basal_nodes;

  auto set_pos = [&](auto data) {
    const auto& ctd = meshSpecs[0]->ctd;
    data->top_side_pos = ctd.side_count-1;
    data->bot_side_pos = ctd.side_count-2;
  };
  set_pos(m_elem_layers_data_gid);
  set_pos(m_elem_layers_data_lid);
  set_pos(m_node_layers_data_gid);
  set_pos(m_node_layers_data_lid);

  // Set layer data in the field accessor
  bool useGlimmerSpacing = m_params->get("Use Glimmer Spacing", false);
  int num_elem_layers = m_elem_layers_data_lid->numLayers;
  int num_node_layers = m_node_layers_data_lid->numLayers;

  std::vector<double> node_layers_coord(num_node_layers);
  if(useGlimmerSpacing) {
    for (int i = 0; i < num_node_layers; i++)
      node_layers_coord[num_elem_layers-i] = 1.0- (1.0 - std::pow(double(i) / num_elem_layers + 1.0, -2))/(1.0 - std::pow(2.0, -2));
  } else {
    //uniform layers
    for (int i = 0; i < num_node_layers; i++)
      node_layers_coord[i] = double(i) / num_elem_layers;
  }

  std::vector<double> elem_layer_thickness(num_elem_layers);
  for (int i = 0; i < num_elem_layers; i++)
    elem_layer_thickness[i] = node_layers_coord[i+1]-node_layers_coord[i];

  auto& vec_states = m_field_accessor->getMeshVectorStates();
  auto& int_states = m_field_accessor->getMeshScalarIntegerStates();
  auto& int64_states = m_field_accessor->getMeshScalarInteger64States();

  vec_states["elem_layer_thickness"] = elem_layer_thickness;
  vec_states["node_layers_coords"] = node_layers_coord;
  int_states["ordering"] = m_elem_layers_data_lid->layerOrd ? 0 : 1;
  int_states["num_layers"] = num_elem_layers;
  int64_states["max_2d_elem_gid"] = m_elem_layers_data_gid->numHorizEntities-1;
  int64_states["max_2d_node_gid"] = m_elem_layers_data_gid->numHorizEntities-1;

  m_bulk_data_set = true;
}

} // namespace Albany
