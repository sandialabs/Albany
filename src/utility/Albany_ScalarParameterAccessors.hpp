#ifndef ALBANY_SCALAR_PARAMETER_ACCESSORS_HPP
#define ALBANY_SCALAR_PARAMETER_ACCESSORS_HPP

#include "Albany_Meta_Utils.hpp"
#include "PHAL_AlbanyTraits.hpp"


namespace Albany
{

template<typename ST>
struct ScalarParameterAccessor {
  ST& getValue () { return value; }
  ST value;
};

template<typename EvalT>
struct ScalarParameterAccessors {
  std::map<std::string,std::shared_ptr<ScalarParameterAccessor<typename EvalT::ScalarT>>> accessors;
};

using AlbanyEvalTypes =
  Albany::TypeList<PHAL::AlbanyTraits::Residual,
                  PHAL::AlbanyTraits::Jacobian,
                  PHAL::AlbanyTraits::Tangent,
                  PHAL::AlbanyTraits::DistParamDeriv,
                  PHAL::AlbanyTraits::HessianVec>;

template<typename EvalT>
using AccessorsRCP = Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>;

using AccessorsValueList = typename ApplyTemplate<AccessorsRCP,AlbanyEvalTypes>::type;

using ScalarParameterAccessorsMap = Albany::TypeMap<AlbanyEvalTypes,AccessorsValueList>;

} // namespace Albany

#endif // ALBANY_SCALAR_PARAMETER_ACCESSORS_HPP