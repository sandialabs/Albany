#include "Phalanx_MDField.hpp"
#include "AAdapt_RC_DataTypes_impl.hpp"
#include "Albany_Layouts.hpp"
#include "AAdapt_RC_Field.hpp"

namespace AAdapt {
namespace rc {

template<int rank> Field<rank>::Field () : valid_(false) {}

template<int rank>
bool Field<rank>::
init (const Teuchos::ParameterList& p, const std::string& name) {
  valid_ = p.isType<std::string>(name + " RC Name");
  if ( ! valid_) return false;
  f_ = typename RTensor<rank>::type(
    p.get<std::string>(name + " RC Name"),
    p.get< Teuchos::RCP<PHX::DataLayout> >(name + " Data Layout"));
  return true;
}

template<int rank>
Field<rank>::operator bool () const { return valid_; }

#define rc_for(i, dim)                                  \
  for (typename RTensor<2>::type::size_type i = 0; i < f_.dimension(dim); ++i)

template<> template<typename ad_type>
void Field<0>::addTo (typename Tensor<ad_type, 0>::type& f_incr) const {
  rc_for(cell, 0) rc_for(qp, 1)
    f_incr(cell,qp) += f_(cell,qp);
}
template<> template<typename ad_type>
void Field<1>::addTo (typename Tensor<ad_type, 1>::type& f_incr) const {
  rc_for(cell, 0) rc_for(qp, 1) rc_for(i0, 2)
    f_incr(cell,qp,i0) += f_(cell,qp,i0);
}
template<> template<typename ad_type>
void Field<2>::addTo (typename Tensor<ad_type, 2>::type& f_incr) const {
  rc_for(cell, 0) rc_for(qp, 1) rc_for(i0, 2) rc_for(i1, 3)
    f_incr(cell,qp,i0,i1) += f_(cell,qp,i0,i1);
}

aadapt_rc_eti_class(Field)
#define eti(ad_type, rank)                                              \
  template void Field<rank>::addTo<ad_type>(Tensor<ad_type, rank>::type&) const;
aadapt_rc_apply_to_all_ad_types_all_ranks(eti)
#undef eti

} // namespace rc
} // namespace AAdapt
