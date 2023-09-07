#include "Albany_OmegahFieldContainer.hpp"

namespace Albany {

OmegahFieldContainer::
OmegahFieldContainer (const Teuchos::RCP<OmegahAbstractMesh>& mesh)
 : m_mesh (mesh)
{
  // Nothing to do here
}

bool OmegahFieldContainer::
has_field (const std::string& name)
{
  return m_fields.find(name)!=m_fields.end();
}

void OmegahFieldContainer::
add_field (const std::string& name,
           const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION (has_field(name), std::runtime_error,
      "Error! Attempt to re-add the same field to this OmegahFieldContainer.\n"
      "  Field name: " + name + "\n");
}

Teuchos::RCP<Thyra_Vector>
OmegahFieldContainer::
get_field (const std::string& name)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!has_field(name), std::runtime_error,
      "Error! Attempt to retrieve a field never added to this OmegahFieldContainer.\n"
      "  Field name: " + name + "\n");
  return m_fields.at(name);
}

} // namespace Albany
