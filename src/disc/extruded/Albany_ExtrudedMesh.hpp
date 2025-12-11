#ifndef ALBANY_EXTRUDED_MESH_HPP
#define ALBANY_EXTRUDED_MESH_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_LayeredMeshNumbering.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_ExtrudedMeshFieldAccessor.hpp"

#include <Teuchos_RCP.hpp>

namespace Albany {

class ExtrudedMesh : public AbstractMeshStruct {
public:
  template<typename T>
  using strmap_t = std::map<std::string,T>;

  ExtrudedMesh (const Teuchos::RCP<AbstractMeshStruct>& basal_mesh,
                const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<const Teuchos_Comm>& comm);

  virtual ~ExtrudedMesh () = default;

  std::string meshLibName () const override {
    return "Albany";
  }

  const Teuchos::RCP<AbstractMeshStruct>& basal_mesh () const { return m_basal_mesh; }

  const Teuchos::RCP<const Teuchos_Comm>& comm() const { return m_comm; }

  LO get_num_local_nodes () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_num_local_nodes.\n");
    return m_basal_mesh->get_num_local_nodes()*layers_data.node.gid->numLayers;
  }
  LO get_num_local_elements () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_num_local_elements.\n");
    return m_basal_mesh->get_num_local_elements()*layers_data.cell.gid->numLayers;
  }
  GO get_max_node_gid () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_max_node_gid.\n");
    return layers_data.node.gid->numHorizEntities*layers_data.node.gid->numLayers;
  }
  GO get_max_elem_gid () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_max_elem_gid.\n");
    return layers_data.cell.gid->numHorizEntities*layers_data.cell.gid->numLayers;
  }

  Teuchos::RCP<AbstractMeshFieldAccessor> get_field_accessor () const override {
    return m_field_accessor;
  }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                     const Teuchos::RCP<StateInfoStruct>& sis,
                     strmap_t<Teuchos::RCP<StateInfoStruct> > side_set_sis) override;

  void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm) override;

  std::string get_basal_part_name (const std::string& part_name) const {
    return m_part_to_basal_part.at(part_name);
  }

protected:

  Teuchos::RCP<const Teuchos_Comm>          m_comm;
  Teuchos::RCP<Teuchos::ParameterList>      m_params;

  Teuchos::RCP<AbstractMeshStruct>          m_basal_mesh;

  Teuchos::RCP<ExtrudedMeshFieldAccessor>   m_field_accessor;

  strmap_t<std::string>                     m_part_to_basal_part;
};

} // namespace Albany

#endif // ALBANY_EXTRUDED_MESH_HPP
