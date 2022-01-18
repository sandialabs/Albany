#ifndef ALBANY_ACCESSORS_HPP
#define ALBANY_ACCESSORS_HPP

#include "Albany_Meta_Utils.hpp"
#include "PHAL_AlbanyTraits.hpp"


namespace Albany
{

template<typename ST>
struct Accessor {
  ST& getValue () { return value; }
  ST value;
};

template<typename EvalT>
struct Accessors {
  std::map<std::string,std::shared_ptr<Accessor<typename EvalT::ScalarT>>> accessors;
};

using AccessorsKeyList =
  Albany::TypeList<PHAL::AlbanyTraits::Residual,
                  PHAL::AlbanyTraits::Jacobian,
                  PHAL::AlbanyTraits::Tangent,
                  PHAL::AlbanyTraits::DistParamDeriv,
                  PHAL::AlbanyTraits::HessianVec>;
using AccessorsValueList =
  Albany::TypeList<Teuchos::RCP<Albany::Accessors<PHAL::AlbanyTraits::Residual>>,
                  Teuchos::RCP<Albany::Accessors<PHAL::AlbanyTraits::Jacobian>>,
                  Teuchos::RCP<Albany::Accessors<PHAL::AlbanyTraits::Tangent>>,
                  Teuchos::RCP<Albany::Accessors<PHAL::AlbanyTraits::DistParamDeriv>>,
                  Teuchos::RCP<Albany::Accessors<PHAL::AlbanyTraits::HessianVec>>>;
using AccessorsMap = Albany::TypeMap<AccessorsKeyList,AccessorsValueList>;

} // namespace Albany

#endif // ALBANY_ACCESSORS_HPP