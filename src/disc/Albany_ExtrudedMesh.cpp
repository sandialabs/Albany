#include "Albany_ExtrudedMesh.hpp"

#include "Albany_DiscretizationUtils.hpp"

namespace Albany {

ExtrudedMesh::
ExtrudedMesh (const Teuchos::RCP<const AbstractMeshStruct>& basal_mesh,
              const Teuchos::RCP<Teuchos::ParameterList>& params,
              const Teuchos::RCP<const Teuchos_Comm>& comm)
 : m_comm (comm)
 , m_basal_mesh (basal_mesh)
{
  // Sanity checks
  TEUCHOS_TEST_FOR_EXCEPTION (basal_mesh.is_null(), std::invalid_argument,
      "[ExtrudedMesh] Error! Invalid basal mesh pointer.\n");

  const auto basal_mesh_specs = m_basal_mesh->meshSpecs[0];
  const int basalNumDim = basal_mesh_specs->numDim;

  TEUCHOS_TEST_FOR_EXCEPTION (basalNumDim<1 or basalNumDim>2, std::logic_error,
      "[ExtrudedMesh] Error! ExtrudedMesh only available in 2D and 3D.\n"
      "  - basal mesh dim: " << basalNumDim << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (params.is_null(), std::runtime_error,
      "[ExtrudedMesh] Error! Invalid parameter list pointer.\n");

  // Create elem layers data
  auto num_layers = params->get<int>("NumLayers");
  TEUCHOS_TEST_FOR_EXCEPTION (num_layers<=0, Teuchos::Exceptions::InvalidParameterValue,
      "[ExtrudedMesh] Error! Number of layers must be strictly positive.\n"
      "  - NumLayers: " << num_layers << "\n");

  auto ordering = params->get("Columnwise Ordering", false)
                ? LayeredMeshOrdering::COLUMN
                : LayeredMeshOrdering::LAYER;

  GO num_lcl_basal_elems = basal_mesh->get_num_local_elements();
  GO num_glb_basal_elems;
  Teuchos::reduceAll(*m_comm,Teuchos::REDUCE_SUM,1,&num_lcl_basal_elems,&num_glb_basal_elems);

  m_elem_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(num_glb_basal_elems,num_layers,ordering));
  m_elem_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_lcl_basal_elems,num_layers,ordering));

  // Create extruded sideSets/nodeSets/elemBlocks names
  std::vector<std::string> nsNames = {"lateral", "bottom", "top"};
  std::vector<std::string> ssNames = {"lateralside", "basalside", "upperside"};
  std::vector<std::string> lateralParts = {"lateralside"};

  for (const auto& ns : basal_mesh_specs->nsNames) {
    nsNames.push_back ("extruded_" + ns);
    nsNames.push_back ("basal" + ns);
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
  std::string elem2d_name(basal_mesh_specs->ctd.base->name);
  std::string tria = shards::getCellTopologyData<shards::Triangle<3> >()->name;
  auto wedge_topo = shards::CellTopology(shards::getCellTopologyData<shards::Wedge<6>>());
  shards::CellTopology elem_topo;
  if (elem2d_name==tria) {
    elem_topo = wedge_topo;
  } else {
    throw Teuchos::Exceptions::InvalidParameterValue(
      "[ExtrudedSTKMeshStruct] Invalid/unsupported basal mesh element type.\n"
      "  - valid element types: " + tria + "\n"
      "  - basal alement type : " + elem2d_name + "\n");
  }
  const CellTopologyData& ctd = *elem_topo.getCellTopologyData();

  // Compute workset size
  int basalWorksetSize = basal_mesh_specs->worksetSize;
  int worksetSizeMax = params->get<int>("Workset Size");
  int ebSizeMaxEstimate = basalWorksetSize * num_layers; // This is ebSizeMax when basalWorksetSize is max
  int worksetSize = computeWorksetSize(worksetSizeMax, ebSizeMaxEstimate);

  // Finally, we can create the mesh specs
  this->meshSpecs.resize(1,Teuchos::rcp(
        new MeshSpecsStruct(MeshType::Extruded, ctd, basalNumDim+1, nsNames, ssNames,
                            worksetSize, ebName, ebNameToIndex)));
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

} // namespace Albany
