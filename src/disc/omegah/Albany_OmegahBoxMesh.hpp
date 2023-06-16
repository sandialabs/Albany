#ifndef ALBANY_OMEGAH_BOX_MESH_HPP
#define ALBANY_OMEGAH_BOX_MESH_HPP

#include "Albany_OmegahAbstractMesh.hpp"

namespace Albany {

template<unsigned Dim>
class OmegahBoxMesh : public OmegahAbstractMesh {
public:
  OmegahBoxMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<const Teuchos_Comm>& comm, const int numParams);

protected:
  Teuchos::RCP<const Teuchos::ParameterList>
  getValidDiscretizationParameters() const;
};

} // namespace Albany

#endif // ALBANY_ABSTRACT_OMEGAH_MESH_HPP
