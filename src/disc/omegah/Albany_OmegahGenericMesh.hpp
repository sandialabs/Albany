#ifndef ALBANY_OMEGAH_GENERIC_MESH_HPP
#define ALBANY_OMEGAH_GENERIC_MESH_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_OmegahMeshFieldAccessor.hpp"

#include "Omega_h_mesh.hpp"

namespace Albany {

class OmegahGenericMesh : public AbstractMeshStruct {
public:
  template<typename T>
  using strmap_t = std::map<std::string,T>;

  virtual ~OmegahGenericMesh () = default;

  // ------------- Override from base class ------------- //
  std::string meshType () const override { return "Omega_h"; }

  Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>&
  getMeshSpecs() override { return m_mesh_specs; }

  const Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>&
  getMeshSpecs() const override { return m_mesh_specs; }


  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& commT,
                     const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                     const unsigned int worksetSize,
                     const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis) override;

  // ------------- Omegah specific methods -------------- //

  Teuchos::RCP<OmegahMeshFieldAccessor> get_field_accessor () const { return m_field_accessor; }

  const Omega_h::Mesh& getOmegahMesh () const { return m_mesh; }
        Omega_h::Mesh& getOmegahMesh ()       { return m_mesh; }

  bool hasRestartSolution () const { return m_has_restart_solution; }

  ViewLR<const double*,DeviceMemSpace> coords_dev  () const { return m_coords_d; }
  ViewLR<const double*,HostMemSpace>   coords_host () const { return m_coords_h; }

  Omega_h::I32 get_ns_tag (const std::string& ns) const { return m_node_sets_tags.at(ns); }

  int part_dim (const std::string& part_name) const;

  // Declare a new part of the mesh, and/or set a tag in the mesh, which states
  // whether each entity belongs to the part or not.
  // If markDownward=true, then all entities of lower dimension bounding each marked entity
  // will be marked as belonging to the part.
  void declare_part (const std::string& name, const Topo_type t);
  void declare_part (const std::string& name, const Topo_type t,
                     Omega_h::Read<Omega_h::I8> is_entity_in_part,
                     const bool markDownward);
  void mark_part_entities (const std::string& name,
                           Omega_h::Read<Omega_h::I8> is_entity_in_part,
                           const bool markDownward);

protected:

  Omega_h::Mesh  m_mesh;

  // We map node/side sets names to an int flag. These tags *MUST* be
  // bitwise exclusive, so that we can use a single int to store many of them.
  // Moreover, they must be strictly positive, so that we can do `a &= tag` to
  // check if the tag is set.
  std::map<std::string,Omega_h::I32>   m_node_sets_tags;
  std::map<std::string,Omega_h::I32>   m_side_sets_tags;

  // Given a part name, returns its topology (in the form of an Omega_h enum
  std::map<std::string,Topo_type>  m_part_topo;

  Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > m_mesh_specs;

  Teuchos::RCP<OmegahMeshFieldAccessor>    m_field_accessor;

  ViewLR<const double*,DeviceMemSpace>  m_coords_d;
  ViewLR<      double*,HostMemSpace>    m_coords_h;

  bool m_has_restart_solution = false;
};

} // namespace Albany

#endif // ALBANY_OMEGAH_GENERIC_MESH_HPP
