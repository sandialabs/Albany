#include "Albany_OmegahOshMesh.hpp"
#include "Albany_Omegah.hpp"
#include "Albany_OmegahUtils.hpp"

#include <Omega_h_file.hpp> // Omega_h::binary::read
#include <Omega_h_mark.hpp> // Omega_h::mark_by_class

namespace Albany {

OmegahOshMesh::
OmegahOshMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<Omega_h::Mesh> mesh, 
               const int /* numParams */)
{
  loadOmegahMesh(params, mesh, 0);
}

OmegahOshMesh::
OmegahOshMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
    const int /* numParams */) {
  auto& lib = get_omegah_lib();
  const auto& filename = params->get<std::string>("Filename");
  auto mesh = Teuchos::RCP<Omega_h::Mesh>(new Omega_h::Mesh(&lib));
  Omega_h::binary::read(filename, lib.world(), mesh.get());
  loadOmegahMesh(params, mesh, 0);
}

void OmegahOshMesh::
loadOmegahMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<Omega_h::Mesh> mesh, 
               const int /* numParams */)
{
  m_mesh = mesh;
  if (params->get("Rebalance",true)) {
    m_mesh->balance(); // re-partition to the number of ranks in world communicator
  }
  m_field_accessor = Teuchos::rcp(new OmegahMeshFieldAccessor(m_mesh));
  m_coords_d = m_mesh->coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);

  // Node/Side sets names
  std::vector<std::string> nsNames, ssNames;

  // The omegah 'exo2osh' converter creates geometric model entities from node
  // and side sets that exist within the exodus file.
  // The mesh entities in the sets are then 'classified' (sets the association)
  // on those model entities.
  // 'Classification' of mesh entities to the geometric model is an alternative
  // to the generic creation of 'parts' (sets of mesh entities with a label) and
  // provides a subset of the functionality.
  // Note, 'classification' is the approach taken when having (at a minimum) a
  // topological definition of the domain is a common part of the mesh
  // generation/adaptation workflow.
  // A dimension and id uniquely defines a geometric model entity.
  const auto& parts_names = params->get<Teuchos::Array<std::string>>("Mark Parts",{});
  for (const auto& pn : parts_names) {
          auto& pl = params->sublist(pn);
    const auto topo_str = pl.get<std::string>("Topo");
    const auto topo = str2topo(pl.get<std::string>("Topo"));
    const int dim = topo_dim(topo);
    const int id  = pl.get<int>("Id");
    const bool markDownward = pl.get<int>("Mark Downward",true); // Is default=true ok?
    auto is_in_part = Omega_h::mark_by_class(m_mesh.get(), dim, dim, id);
    this->declare_part(pn,topo,is_in_part,markDownward);

    if (dim==0) {
      nsNames.push_back(pn);
    } else if (dim==m_mesh->dim()-1) {
      ssNames.push_back(pn);
    }
  }

  const CellTopologyData* ctd;

  TEUCHOS_TEST_FOR_EXCEPTION (
      m_mesh->family()!=Omega_h_Family::OMEGA_H_SIMPLEX and m_mesh->family()!=Omega_h_Family::OMEGA_H_HYPERCUBE,
      std::runtime_error,
      "Error! OmegahOshMesh only available for simplex/hypercube meshes.\n");

  Topo_type elem_topo;
  switch (m_mesh->dim()) {
    case 1:
      ctd = shards::getCellTopologyData<shards::Line<2>>();
      elem_topo = Topo_type::edge;
      break;
    case 2:
      if (m_mesh->family()==Omega_h_Family::OMEGA_H_SIMPLEX) {
        ctd = shards::getCellTopologyData<shards::Triangle<3>>();
        elem_topo = Topo_type::triangle;
      } else {
        ctd = shards::getCellTopologyData<shards::Quadrilateral<4>>();
        elem_topo = Topo_type::quadrilateral;
      }
      break;
    case 3:
      if (m_mesh->family()==Omega_h_Family::OMEGA_H_SIMPLEX) {
        ctd = shards::getCellTopologyData<shards::Tetrahedron<4>>();
        elem_topo = Topo_type::tetrahedron;
      } else {
        ctd = shards::getCellTopologyData<shards::Hexahedron<8>>();
        elem_topo = Topo_type::hexahedron;
      }
      break;
  }
  std::string ebName = "element_block_0";
  std::map<std::string,int> ebNameToIndex = 
  {
    { ebName, 0}
  };
  this->declare_part(ebName,elem_topo);

  // Omega_h does not know what worksets are, so all elements are in one workset
  this->meshSpecs.resize(1);
  this->meshSpecs[0] = Teuchos::rcp(
      new MeshSpecsStruct(MeshType::Unstructured, *ctd, m_mesh->dim(),
                          nsNames, ssNames, m_mesh->nelems(), ebName,
                          ebNameToIndex));
}

} // namespace Albany

