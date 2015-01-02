#include "Phalanx_MDField.hpp"
#include "Intrepid_MiniTensor_Tensor.h"
#include "AAdapt_RC_DataTypes_impl.hpp"
#include "Albany_Layouts.hpp"
#include "AAdapt_RC_Field.hpp"

namespace AAdapt {
namespace rc {

template<int rank> Field<rank>::Field () : valid_(false) {}

template<int rank>
bool Field<rank>::
init (const Teuchos::ParameterList& p, const std::string& name) {
  const std::string name_rc = name + " RC Name";
  valid_ = p.isType<std::string>(name_rc);
  if ( ! valid_) return false;
  f_ = typename RTensor<rank>::type(
    p.get<std::string>(name_rc),
    p.get< Teuchos::RCP<PHX::DataLayout> >(name + " Data Layout"));
  return true;
}

template<int rank>
Field<rank>::operator bool () const { return valid_; }

#define loop(f, i, dim)                                                 \
  for (typename RTensor<2>::type::size_type i = 0; i < f.dimension(dim); ++i)
#define loopf(i, dim) loop(f_, i, dim)

template<> template<typename ad_type>
void Field<0>::addTo (typename Tensor<ad_type, 0>::type& f_incr) const {
  loopf(cell, 0) loopf(qp, 1)
    f_incr(cell,qp) += f_(cell,qp);
}
template<> template<typename ad_type>
void Field<1>::addTo (typename Tensor<ad_type, 1>::type& f_incr) const {
  loopf(cell, 0) loopf(qp, 1) loopf(i0, 2)
    f_incr(cell,qp,i0) += f_(cell,qp,i0);
}
template<> template<typename ad_type>
void Field<2>::addTo (typename Tensor<ad_type, 2>::type& f_incr) const {
  loopf(cell, 0) loopf(qp, 1) loopf(i0, 2) loopf(i1, 3)
    f_incr(cell,qp,i0,i1) += f_(cell,qp,i0,i1);
}

template<> template<typename ad_type>
void Field<2>::multiplyInto (typename Tensor<ad_type, 2>::type& f_incr) const {
  Intrepid::Tensor<ad_type, 3> f_incr_mt(f_.dimension(2));
  Intrepid::Tensor<RealType, 3> f_accum_mt(f_.dimension(2));
  loopf(cell, 0) loopf(qp, 1) {
    loopf(i0, 2) loopf(i1, 3) f_incr_mt(i0, i1) = f_incr(cell, qp, i0, i1);
    loopf(i0, 2) loopf(i1, 3) f_accum_mt(i0, i1) = f_(cell, qp, i0, i1);
    Intrepid::Tensor<ad_type, 3> C = Intrepid::dot(f_incr_mt, f_accum_mt);
    loopf(i0, 2) loopf(i1, 3) f_incr(cell, qp, i0, i1) = C(i0, i1);
  }
}

#undef loopf
#undef loop

aadapt_rc_eti_class(Field)
#define eti(ad_type, rank)                                              \
  template void Field<rank>::addTo<ad_type>(Tensor<ad_type, rank>::type&) const;
aadapt_rc_apply_to_all_ad_types_all_ranks(eti)
#undef eti
#define eti(ad_type, arg2)                                              \
  template void Field<2>::multiplyInto<ad_type>(Tensor<ad_type,2>::type&) const;
aadapt_rc_apply_to_all_ad_types(eti,)
#undef eti

} // namespace rc
} // namespace AAdapt
