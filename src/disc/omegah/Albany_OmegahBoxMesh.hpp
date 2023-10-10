#ifndef ALBANY_OMEGAH_BOX_MESH_HPP
#define ALBANY_OMEGAH_BOX_MESH_HPP

#include "Albany_OmegahAbstractMesh.hpp"

namespace Albany {

template<unsigned Dim>
class OmegahBoxMesh : public OmegahAbstractMesh {
public:
  OmegahBoxMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<const Teuchos_Comm>& comm, const int numParams);

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& commT,
                     const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                     const unsigned int worksetSize,
                     const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis) override
  {
    printf("TODO: add code to setup data to store states to mesh.\n");
  }
  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& commT,
                    const Teuchos::RCP<StateInfoStruct>& sis,
                    const unsigned int worksetSize,
                    const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis) override;

protected:

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidDiscretizationParameters() const;
};

} // namespace Albany

#endif // ALBANY_ABSTRACT_OMEGAH_MESH_HPP
