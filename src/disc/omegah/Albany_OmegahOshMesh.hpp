#ifndef ALBANY_OMEGAH_OSH_MESH_HPP
#define ALBANY_OMEGAH_OSH_MESH_HPP

#include "Albany_OmegahGenericMesh.hpp"

namespace Albany {

class OmegahOshMesh : public OmegahGenericMesh
{
public:
  OmegahOshMesh (const Teuchos::RCP<Teuchos::ParameterList>& params);

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm) override;
};

} // namespace Albany

#endif // ALBANY_OMEGAH_OSH_MESH_HPP
