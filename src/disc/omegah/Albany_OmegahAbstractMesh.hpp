#ifndef ALBANY_ABSTRACT_OMEGAH_MESH_HPP
#define ALBANY_ABSTRACT_OMEGAH_MESH_HPP

#include "Albany_AbstractMeshStruct.hpp"

#include "Omega_h_mesh.hpp"

namespace Albany {

class OmegahAbstractMesh : public AbstractMeshStruct {
public:
  std::string meshType () const override { return "Omega_h"; }

  const Omega_h::Mesh& getOmegahMesh () const { return m_mesh; }
        Omega_h::Mesh& getOmegahMesh ()       { return m_mesh; }

  Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>&
  getMeshSpecs() override { return m_mesh_specs; }

  bool hasRestartSolution () const { return m_has_restart_solution; }

  const Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>&
  getMeshSpecs() const override { return m_mesh_specs; }
protected:

  Omega_h::Mesh  m_mesh;

  Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > m_mesh_specs;

  bool m_has_restart_solution = false;
};

} // namespace Albany

#endif // ALBANY_ABSTRACT_OMEGAH_MESH_HPP
