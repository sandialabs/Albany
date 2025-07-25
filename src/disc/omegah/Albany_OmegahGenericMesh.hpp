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

  OmegahGenericMesh(const Teuchos::RCP<Teuchos::ParameterList>& params);

  virtual ~OmegahGenericMesh () = default;

  // ------------- Override from base class ------------- //
  std::string meshLibName () const override { return "Omega_h"; }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm) override;
  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm) override;

  LO get_num_local_nodes () const override;
  LO get_num_local_elements  () const override;
  GO get_max_node_gid () const override;
  GO get_max_elem_gid () const override;

  Teuchos::RCP<AbstractMeshFieldAccessor> get_field_accessor () const override { return m_field_accessor; }

  // ------------- Omegah specific methods -------------- //

        Teuchos::RCP<const Omega_h::Mesh>  getOmegahMesh () const { return m_mesh.getConst(); }
  const Teuchos::RCP<      Omega_h::Mesh>& getOmegahMesh ()       { return m_mesh; }

  bool hasRestartSolution () const { return m_has_restart_solution; }

  DeviceView1d<const double>             coords_dev  () const { return m_coords_d; }
  DeviceView1d<      double>::HostMirror coords_host () const { return m_coords_h; }

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

  void reset_mesh (const Teuchos::RCP<Omega_h::Mesh> mesh);

  void setCoordinates();
  std::vector<std::string> createNodeSets();
  std::vector<std::string> createSideSets();

protected:
  // Load a mesh from a .osh file
  void loadOmegahMesh ();

  // Builds a box mesh
  void buildBox (const int dim);

  struct GeoModelEntity {
    int dim;
    int id;
  };

  using PartToGeoModelEntities = strmap_t<std::vector<GeoModelEntity>>;
  PartToGeoModelEntities setNodeSetsGeoModelEntities () const;
  PartToGeoModelEntities setSideSetsGeoModelEntities () const;

  void loadRequiredInputFields (const Teuchos::RCP<const Teuchos_Comm>& comm);

  Teuchos::RCP<Teuchos::ParameterList> m_params;

  Teuchos::RCP<Omega_h::Mesh>  m_mesh;

  // Given a part name, returns its topology (in the form of an Omega_h enum
  std::map<std::string,Topo_type>  m_part_topo;

  Teuchos::RCP<OmegahMeshFieldAccessor>    m_field_accessor;

  DeviceView1d<const double>                m_coords_d;
  DeviceView1d<      double>::HostMirror    m_coords_h;

  mutable GO m_max_node_gid = -1; //set when get_max_node_gid() called
  mutable GO m_max_elem_gid = -1; //set when get_max_elem_gid() called

  bool m_has_restart_solution = false;

private:

  // For each node/side set name, store the geometric model entities that belong to it
  PartToGeoModelEntities nodeSetsGeoModelEntities;
  PartToGeoModelEntities sideSetsGeoModelEntities;
};

} // namespace Albany

#endif // ALBANY_OMEGAH_GENERIC_MESH_HPP
