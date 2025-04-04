#ifndef ALBANY_OMEGAH_BOX_MESH_HPP
#define ALBANY_OMEGAH_BOX_MESH_HPP

#include "Albany_OmegahGenericMesh.hpp"

namespace Albany {

template<unsigned Dim>
class OmegahBoxMesh : public OmegahGenericMesh {
public:
  static_assert (Dim>=1 && Dim<=3, "Unsupported dimension for OmegahBoxMesh");

  OmegahBoxMesh (const Teuchos::RCP<Teuchos::ParameterList>& params);

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm) override;

  Omega_h::Read<Omega_h::I8>
  create_ns_tag (const std::string& name,
                 const int comp,
                 const double tgt_value) const;
protected:

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidDiscretizationParameters() const;
};

} // namespace Albany

#endif // ALBANY_ABSTRACT_OMEGAH_MESH_HPP
