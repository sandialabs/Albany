#ifndef ALBANY_OMEGAH_OSH_MESH_HPP
#define ALBANY_OMEGAH_OSH_MESH_HPP

#include "Albany_OmegahGenericMesh.hpp"

namespace Albany {

class OmegahOshMesh : public OmegahGenericMesh
{
public:
  OmegahOshMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<const Teuchos_Comm>& comm, const int numParams);

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& commT,
                    const Teuchos::RCP<StateInfoStruct>& sis,
                    const unsigned int worksetSize,
                    const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis) override
  {
    throw NotYetImplemented("OmegahOshMesh::setBulkData");
  }
};

} // namespace Albany

#endif // ALBANY_OMEGAH_OSH_MESH_HPP
