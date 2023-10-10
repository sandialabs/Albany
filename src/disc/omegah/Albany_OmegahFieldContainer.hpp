#ifndef ALBANY_OMEGAH_FIELD_CONTAINER_HPP
#define ALBANY_OMEGAH_FIELD_CONTAINER_HPP

#include "Albany_OmegahAbstractMesh.hpp"
#include "Albany_ThyraTypes.hpp"

#include <Teuchos_RCP.hpp>

#include <string>
#include <map>

namespace Albany {

// Note: this class is nothing but a glorified std::map,
// which we could have just added to OmegahDiscretization.
// I created the class in case we later decide to store the
// fields inside the omegah mesh, in which case this class
// will simply be an interface between libraries. If we never
// do such change, we can get rid of this class, and store
// everything inside the discretization object.

class OmegahFieldContainer
{
public:
  template<typename T>
  using strmap_t = std::map<std::string,T>;

  OmegahFieldContainer (const Teuchos::RCP<OmegahAbstractMesh>& mesh);

  bool has_field (const std::string& name);

  void add_field (const std::string& name,
                  const Teuchos::RCP<const Thyra_VectorSpace>& vs);

  Teuchos::RCP<Thyra_Vector> get_field (const std::string& name);

  const StateInfoStruct& get_nodal_sis () const { return m_nodal_sis; }
protected:

  StateInfoStruct m_nodal_sis;

  Teuchos::RCP<OmegahAbstractMesh>        m_mesh;
  strmap_t<Teuchos::RCP<Thyra_Vector>>    m_fields;
};

} // namespace Albany

#endif // ALBANY_OMEGAH_FIELD_CONTAINER_HPP
