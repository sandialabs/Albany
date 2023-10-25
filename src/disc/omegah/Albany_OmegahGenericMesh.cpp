#include "Albany_OmegahGenericMesh.hpp"

namespace Albany
{

void OmegahGenericMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& commT,
              const Teuchos::RCP<StateInfoStruct>& sis,
              const unsigned int worksetSize,
              const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis)
{
  m_field_accessor = Teuchos::rcp(new OmegahMeshFieldAccessor(m_mesh));
  if (not sis.is_null()) {
    m_field_accessor->addStateStructs (sis);
  }
}

} // namespace Albany
