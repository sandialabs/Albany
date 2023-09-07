#include "Albany_OmegahDiscretization.hpp"

namespace Albany {

OmegahDiscretization::
OmegahDiscretization(
  const Teuchos::RCP<Teuchos::ParameterList>& discParams,
  const int                                   neq,
  Teuchos::RCP<OmegahAbstractMesh>&           mesh,
  const Teuchos::RCP<const Teuchos_Comm>&     comm,
  const Teuchos::RCP<RigidBodyModes>& /* rigidBodyModes */,
  const std::map<int, std::vector<std::string>>& /* sideSetEquations */)
 : m_disc_params (discParams)
 , m_mesh_struct(mesh)
 , m_comm (comm)
 , m_neq (neq)
{
  m_num_time_deriv = m_disc_params->get("Number Of Time Derivatives",0);
}

void OmegahDiscretization::
updateMesh () {
  printf ("TODO: add creation of dof managers.\n  Also, change name to the method?\n");
}

void OmegahDiscretization::
setFieldData(const Teuchos::RCP<StateInfoStruct>& sis) {
  printf ("TODO: add code to save states in disc field container, if needed.\n");
}

}  // namespace Albany
