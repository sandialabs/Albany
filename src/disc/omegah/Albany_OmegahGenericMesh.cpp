#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahUtils.hpp"

namespace Albany
{

void OmegahGenericMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& sis)
{
  m_field_accessor = Teuchos::rcp(new OmegahMeshFieldAccessor(m_mesh));
  if (not sis.is_null()) {
    m_field_accessor->addStateStructs (sis);
  }
}

int OmegahGenericMesh::
part_dim (const std::string& part_name) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_part_topo.find(part_name)==m_part_topo.end(), std::runtime_error,
      "[OmegahGenericMesh::part_dim] Error! Cannot find input part: " << part_name << "\n");
  return topo_dim(m_part_topo.at(part_name));
}

void OmegahGenericMesh::
declare_part (const std::string& name, const Topo_type topo)
{
  TEUCHOS_TEST_FOR_EXCEPTION (
      m_part_topo.find(name)!=m_part_topo.end() and m_part_topo[name]!=topo, std::logic_error,
      "[OmegahGenericMesh::declare_part] Error! Redefining part topology to a different value.\n"
      "  - part name: " << name << "\n"
      "  - curr topo: " << e2str(m_part_topo[name]) << "\n"
      "  - new  topo: " << e2str(topo) << "\n");

  // Check that this topo matches the topo of one of the element sub-entities topos
  TEUCHOS_TEST_FOR_EXCEPTION (not m_mesh->has_ents(topo_dim(topo)), std::logic_error,
      "[OmegahGenericMesh::declare_part] Mesh does not store any entity of the input topology.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << topo_dim(topo) << "\n"
      "  - part topo: " << e2str(topo) << "\n"
      "  - mesh dim : " << m_mesh->dim() << "\n"
      "  - mesh type: " << e2str(m_mesh->family()) << "\n");

  // All good, store it
  m_part_topo[name] = topo;
}

void OmegahGenericMesh::
declare_part (const std::string& name, const Topo_type topo,
              Omega_h::Read<Omega_h::I8> is_entity_in_part,
              const bool markDownward)
{
  declare_part (name,topo);
  mark_part_entities (name,is_entity_in_part,markDownward);
}

void OmegahGenericMesh::
mark_part_entities (const std::string& name,
                   Omega_h::Read<Omega_h::I8> is_entity_in_part,
                   const bool markDownward)
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_part_topo.find(name)==m_part_topo.end(), std::runtime_error,
      "[OmegahGenericMesh::set_part_entities] Error! Part not found.\n"
      "  - part name: " << name << "\n");

  auto dim = topo_dim(m_part_topo[name]);

  TEUCHOS_TEST_FOR_EXCEPTION (is_entity_in_part.size()!=m_mesh->nents(dim), std::logic_error,
      "[OmegahGenericMesh::set_part_entities] Error! Input array has the wrong dimensions.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << dim << "\n"
      "  - num ents : " << m_mesh->nents(dim) << "\n"
      "  - array dim: " << is_entity_in_part.size() << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (m_mesh->has_tag(dim,name), std::runtime_error,
      "[OmegahGenericMesh::set_part_entities] Error! A tag with this name was already set.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << dim << "\n");

  m_mesh->add_tag(dim,name,1,is_entity_in_part);

  if (markDownward) {
    TEUCHOS_TEST_FOR_EXCEPTION (dim==0, std::logic_error,
      "[OmegahGenericMesh::set_part_entities] Error! Cannot mark downward if the part dimension is 0.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << dim << "\n")
    Omega_h::Write<Omega_h::I8> downMarked;

    // NOTE: In the following, we keep converting Topo_type to its integer dim. That's because
    //       Omega_h::Mesh seems to store a valid nent(dim) count, but uninited nent(topo) count.
    //       If topo is 2d (or less), we can safely use nent(dim), since there is only one possible
    //       down_topo (edge or vertex) for each topo. But if this method is called to mark a 3d
    //       region, then down ents may be of multiple topos (e.g., tria and quad for a wedge).
    //       Until then, turn topo into dim
    auto upMarked = is_entity_in_part;
    auto topo = m_part_topo[name];
    while (topo!=Topo_type::vertex) {
      const auto down_topo = get_side_topo(topo);

      Omega_h::Write<Omega_h::I8> downMarked (m_mesh->nents(topo_dim(down_topo)),0);

      const int deg = Omega_h::element_degree(topo,down_topo);

      auto adj = m_mesh->ask_down(topo_dim(topo),topo_dim(down_topo));
      auto f = OMEGA_H_LAMBDA (LO i) {
        if (upMarked[i]) {
          for (int j=0; j<deg; ++j) {
            // Note: this has a race condition, but it doesn't matter,
            //       since all threads would write 1
            downMarked[adj.ab2b[i*deg + j]] = 1;
          }
        }
      };
      Kokkos::parallel_for ("OmegahGenericMesh::set_part_entities::markDownward",m_mesh->nents(topo_dim(topo)),f);
      m_mesh->add_tag(topo_dim(down_topo),name,1,read(downMarked));
      upMarked = downMarked;

      topo = down_topo;
    }
  }
}

} // namespace Albany
