#ifndef ALBANY_EXTRUDED_MESH_HPP
#define ALBANY_EXTRUDED_MESH_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_LayeredMeshNumbering.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include <Teuchos_RCP.hpp>

namespace Albany {

class ExtrudedMesh : public AbstractMeshStruct {
public:
  ExtrudedMesh (const Teuchos::RCP<const AbstractMeshStruct>& basal_mesh,
                const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<const Teuchos_Comm>& comm);

  virtual ~ExtrudedMesh () = default;

  std::string meshLibName () const override {
    return "Albany";
  }

  // Checks that the extruded part name is "extruded_XYZ", and return XYZ
  std::string get_basal_part_name (const std::string& extruded_part_name) const;

  const Teuchos::RCP<LayeredMeshNumbering<GO>>&
  layers_data_gid () const { return m_elem_layers_data_gid; }
  const Teuchos::RCP<LayeredMeshNumbering<LO>>&
  layers_data_lid () const { return m_elem_layers_data_lid; }

  const Teuchos::RCP<const AbstractMeshStruct>& basal_mesh () const { return m_basal_mesh; }

  const Teuchos::RCP<const Teuchos_Comm>& comm() const { return m_comm; }

  LO get_num_local_elements () const override {
    return m_basal_mesh->get_num_local_elements()*m_elem_layers_data_gid->numLayers;
  }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                     const Teuchos::RCP<StateInfoStruct>& sis) override
  {
    throw NotYetImplemented("ExtrudedMesh::setFieldData");
  }

  void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm) override
  {
    throw NotYetImplemented("ExtrudedMesh::setBulkData");
  }
protected:

  Teuchos::RCP<const Teuchos_Comm>          m_comm;

  Teuchos::RCP<const AbstractMeshStruct>    m_basal_mesh;

  // ElemShapeType         m_elem_shape;
  Teuchos::RCP<LayeredMeshNumbering<GO>>    m_elem_layers_data_gid;
  Teuchos::RCP<LayeredMeshNumbering<LO>>    m_elem_layers_data_lid;
};

} // namespace Albany

#endif // ALBANY_EXTRUDED_MESH_HPP
