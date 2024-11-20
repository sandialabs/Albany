#ifndef ALBANY_EXTRUDED_MESH_HPP
#define ALBANY_EXTRUDED_MESH_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_LayeredMeshNumbering.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include <Teuchos_RCP.hpp>

namespace Albany {

class ExtrudedMesh : public AbstractMeshStruct {
public:
  ExtrudedMesh (const Teuchos::RCP<AbstractMeshStruct>& basal_mesh,
                const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<const Teuchos_Comm>& comm);

  virtual ~ExtrudedMesh () = default;

  std::string meshLibName () const override {
    return "Albany";
  }

  // Checks that the extruded part name is "extruded_XYZ", and return XYZ
  std::string get_basal_part_name (const std::string& extruded_part_name) const;

  const Teuchos::RCP<LayeredMeshNumbering<GO>>&
  cell_layers_gid () const { return m_elem_layers_data_gid; }
  const Teuchos::RCP<LayeredMeshNumbering<LO>>&
  cell_layers_lid () const { return m_elem_layers_data_lid; }

  const Teuchos::RCP<LayeredMeshNumbering<GO>>&
  node_layers_gid () const { return m_node_layers_data_gid; }
  const Teuchos::RCP<LayeredMeshNumbering<LO>>&
  node_layers_lid () const { return m_node_layers_data_lid; }

  const Teuchos::RCP<AbstractMeshStruct>& basal_mesh () const { return m_basal_mesh; }

  const Teuchos::RCP<const Teuchos_Comm>& comm() const { return m_comm; }

  LO get_num_local_nodes () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_num_local_nodes.\n");
    return m_basal_mesh->get_num_local_nodes()*m_node_layers_data_gid->numLayers;
  }
  LO get_num_local_elements () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_num_local_elements.\n");
    return m_basal_mesh->get_num_local_elements()*m_elem_layers_data_gid->numLayers;
  }
  GO get_max_node_gid () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_max_node_gid.\n");
    return m_node_layers_data_gid->numHorizEntities*m_node_layers_data_gid->numLayers;
  }
  GO get_max_elem_gid () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_max_elem_gid.\n");
    return m_elem_layers_data_gid->numHorizEntities*m_elem_layers_data_gid->numLayers;
  }

  Teuchos::RCP<AbstractMeshFieldAccessor> get_field_accessor () const override {
    return m_basal_mesh->get_field_accessor();
  }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                     const Teuchos::RCP<StateInfoStruct>& sis) override;

  void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm) override;

protected:

  Teuchos::RCP<const Teuchos_Comm>          m_comm;
  Teuchos::RCP<Teuchos::ParameterList>      m_params;

  Teuchos::RCP<AbstractMeshStruct>          m_basal_mesh;

  Teuchos::RCP<LayeredMeshNumbering<GO>>    m_elem_layers_data_gid;
  Teuchos::RCP<LayeredMeshNumbering<LO>>    m_elem_layers_data_lid;
  Teuchos::RCP<LayeredMeshNumbering<GO>>    m_node_layers_data_gid;
  Teuchos::RCP<LayeredMeshNumbering<LO>>    m_node_layers_data_lid;
};

} // namespace Albany

#endif // ALBANY_EXTRUDED_MESH_HPP
