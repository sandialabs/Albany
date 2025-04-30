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

  // Create elem layers data
  auto num_layers = m_params->get<int>("NumLayers");
  TEUCHOS_TEST_FOR_EXCEPTION (num_layers<=0, Teuchos::Exceptions::InvalidParameterValue,
      "[ExtrudedMesh] Error! Number of layers must be strictly positive.\n"
      "  - NumLayers: " << num_layers << "\n");

  // Create extruded sideSets/nodeSets/elemBlocks names
  std::vector<std::string> nsNames = {"lateral", "bottom", "top"};
  std::vector<std::string> ssNames = {"lateralside", "basalside", "upperside"};
  std::vector<std::string> lateralParts = {"lateralside"};

  for (const auto& ns : basal_mesh_specs->nsNames) {
    nsNames.push_back ("extruded_" + ns);
    nsNames.push_back ("basal_" + ns);
  }
  for (const auto& ss : basal_mesh_specs->ssNames) {
    auto pname = "extruded_" + ss;
    ssNames.push_back (pname);
    lateralParts.push_back(pname);
  }
  std::string ebName = "extruded_" + basal_mesh_specs->ebName;
  std::map<std::string,int> ebNameToIndex =
  {
    { ebName, 0}
  };

  // Determine topology
  auto basal_topo = basal_mesh_specs->ctd;
  auto tria_topo = *shards::getCellTopologyData<shards::Triangle<3> >();
  auto quad_topo = *shards::getCellTopologyData<shards::Quadrilateral<4>>();
  auto wedge_topo = *shards::getCellTopologyData<shards::Wedge<6>>();
  CellTopologyData elem_topo, lat_topo;
  if (basal_topo.name==tria_topo.name) {
    elem_topo = wedge_topo;
    lat_topo  = quad_topo;
  } else {
    throw Teuchos::Exceptions::InvalidParameterValue(
      "[ExtrudedSTKMeshStruct] Invalid/unsupported basal mesh element type.\n"
      "  - valid element types: " + std::string(tria_topo.name) + "\n"
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

  // At this point, we cannot assume there will be a discretization on upper/lateral sides,
  // so create "empty" mesh specs, just setting the cell topology and mesh dim. IF a side disc
  // is created, these will be overwritten

  auto& upper_ms = ss_ms["upperside"];
  upper_ms.resize(1, Teuchos::rcp(new MeshSpecsStruct()));
  upper_ms[0]->numDim = basal_topo.dimension;
  upper_ms[0]->ctd = basal_topo;

  auto& lateral_ms = ss_ms["lateralside"];
  lateral_ms.resize(1, Teuchos::rcp(new MeshSpecsStruct()));
  lateral_ms[0]->numDim = lat_topo.dimension;
  lateral_ms[0]->ctd = lat_topo;
}

std::string ExtrudedMesh::
get_basal_part_name (const std::string& extruded_part_name) const
{
  const std::string prefix = "extruded_";
  TEUCHOS_TEST_FOR_EXCEPTION (extruded_part_name.substr(0,prefix.length())!=prefix, std::logic_error,
      "Error! Extruded part name does not start with 'extruded_'.\n"
      " - part name: " + extruded_part_name + "\n");

  return extruded_part_name.substr(prefix.length());
}

void ExtrudedMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& /* sis */)
{
  // Register surface height and mesh thickness in the 2d mesh
  std::string thickness_name = m_params->get<std::string>("Thickness Field Name","thickness");
  std::string surface_height_name = m_params->get<std::string>("Surface Height Field Name","surface_height");
  auto mesh_sis = Teuchos::rcp(new StateInfoStruct());
  auto NDTEN = StateStruct::MeshFieldEntity::NodalDataToElemNode;

  StateStruct::FieldDims dims = {
    static_cast<PHX::DataLayout::size_type>(m_basal_mesh->meshSpecs[0]->worksetSize),
    m_basal_mesh->meshSpecs[0]->ctd.node_count
  };
  mesh_sis->emplace_back(Teuchos::rcp(new StateStruct(surface_height_name,NDTEN,dims,"")));
  mesh_sis->emplace_back(Teuchos::rcp(new StateStruct(thickness_name,NDTEN,dims,"")));
  m_basal_mesh->setFieldData(comm,mesh_sis);
}

void ExtrudedMesh::
setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  if (not m_basal_mesh->isBulkDataSet()) {
    m_basal_mesh->setBulkData(comm);
  }

  // Create layer data structures
  const auto max_basal_node_gid = m_basal_mesh->get_max_node_gid();
  const auto num_basal_nodes    = m_basal_mesh->get_num_local_nodes();
  const auto max_basal_elem_gid = m_basal_mesh->get_max_elem_gid();
  const auto num_basal_elems    = m_basal_mesh->get_num_local_elements();

  const auto num_layers = m_params->get<int>("NumLayers");
  const auto ordering = m_params->get("Columnwise Ordering", false)
                      ? LayeredMeshOrdering::COLUMN
                      : LayeredMeshOrdering::LAYER;

  m_elem_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(max_basal_elem_gid+1,num_layers,ordering));
  m_elem_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_basal_elems,num_layers,ordering));
  m_node_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(max_basal_node_gid+1,num_layers+1,ordering));
  m_node_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_basal_nodes,num_layers+1,ordering));

  auto set_pos = [&](auto data) {
    const auto& ctd = meshSpecs[0]->ctd;
    data->top_side_pos = ctd.side_count-1;
    data->bot_side_pos = ctd.side_count-2;
  };
  set_pos(m_elem_layers_data_gid);
  set_pos(m_elem_layers_data_lid);
  set_pos(m_node_layers_data_gid);
  set_pos(m_node_layers_data_lid);

  m_bulk_data_set = true;
}

} // namespace Albany
