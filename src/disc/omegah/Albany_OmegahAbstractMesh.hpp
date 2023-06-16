#ifndef ALBANY_ABSTRACT_OMEGAH_MESH_HPP
#define ALBANY_ABSTRACT_OMEGAH_MESH_HPP

#include "Albany_AbstractMeshStruct.hpp"

namespace Albany {

class OmegahAbstractMesh : public AbstractMeshStruct {
public:
  std::string meshType () const override { return "Omega_h"; }

        Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>& getMeshSpecs()       override { return m_mesh_specs; }
  const Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>& getMeshSpecs() const override { return m_mesh_specs; }
protected:

  Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > m_mesh_specs;
};

} // namespace Albany

#endif // ALBANY_ABSTRACT_OMEGAH_MESH_HPP
