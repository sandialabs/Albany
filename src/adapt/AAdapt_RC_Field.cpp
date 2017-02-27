//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_MDField.hpp"
#include "MiniTensor_Tensor.h"
#include "AAdapt_RC_DataTypes_impl.hpp"
#include "Albany_Layouts.hpp"
#include "AAdapt_RC_Manager.hpp"
#include "AAdapt_RC_Field.hpp"

namespace AAdapt {
namespace rc {

template<int rank> Field<rank>::Field () : valid_(false) {}

template<int rank>
bool Field<rank>::
init (const Teuchos::ParameterList& p, const std::string& name) {
  const std::string
    name_rc = Manager::decorate(name),
    name_rc_name = name_rc + " Name";
  valid_ = p.isType<std::string>(name_rc_name);
  if ( ! valid_) return false;
  f_ = decltype(f_)(
    p.get<std::string>(name_rc_name),
    p.get< Teuchos::RCP<PHX::DataLayout> >(name_rc + " Data Layout"));
  return true;
}

template<int rank>
Field<rank>::operator bool () const { return valid_; }

template <typename T>
struct SizeType {
  using T_noref = typename std::remove_reference<T>::type;
  using T_noref_nocv = typename std::remove_cv<T_noref>::type;
  using type = typename T_noref_nocv::size_type;
};

#define loop(f, i, dim)                                                 \
  for (typename SizeType<decltype(f)>::type i = 0; i < f.dimension(dim); ++i)
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
void Field<0>::addTo (typename Tensor<ad_type, 0>::type& f_incr,
                      const std::size_t cell, const std::size_t qp) const {
    f_incr(cell,qp) += f_(cell,qp);
}
template<> template<typename ad_type>
void Field<1>::addTo (typename Tensor<ad_type, 1>::type& f_incr,
                      const std::size_t cell, const std::size_t qp) const {
  loopf(i0, 2)
    f_incr(cell,qp,i0) += f_(cell,qp,i0);
}
template<> template<typename ad_type>
void Field<2>::addTo (typename Tensor<ad_type, 2>::type& f_incr,
                      const std::size_t cell, const std::size_t qp) const {
  loopf(i0, 2) loopf(i1, 3)
    f_incr(cell,qp,i0,i1) += f_(cell,qp,i0,i1);
}

namespace {
template<typename ad_type> struct MultiplyWork {
  minitensor::Tensor<ad_type> f_incr_mt;
  minitensor::Tensor<RealType> f_accum_mt;
  MultiplyWork(const std::size_t dim) : f_incr_mt(dim), f_accum_mt(dim) {}
};

template<typename ad_type>
inline void
multiplyIntoImpl (
  const Tensor<const RealType, 2>::type& f_, typename Tensor<ad_type, 2>::type& f_incr,
  const std::size_t cell, const std::size_t qp, MultiplyWork<ad_type>& w)
{
  loopf(i0, 2) loopf(i1, 3) w.f_incr_mt(i0, i1) = f_incr(cell, qp, i0, i1);
  loopf(i0, 2) loopf(i1, 3) w.f_accum_mt(i0, i1) = f_(cell, qp, i0, i1);
  minitensor::Tensor<ad_type> C = minitensor::dot(w.f_incr_mt, w.f_accum_mt);
  loopf(i0, 2) loopf(i1, 3) f_incr(cell, qp, i0, i1) = C(i0, i1);  
}
} // namespace

template<> template<typename ad_type>
void Field<2>::
multiplyInto (typename Tensor<ad_type, 2>::type& f_incr,
              const std::size_t cell, const std::size_t qp) const {
  MultiplyWork<ad_type> w(f_.dimension(2));
  multiplyIntoImpl(f_, f_incr, cell, qp, w);
}
template<> template<typename ad_type>
void Field<2>::multiplyInto (typename Tensor<ad_type, 2>::type& f_incr) const {
  MultiplyWork<ad_type> w(f_.dimension(2));
  loopf(cell, 0) loopf(qp, 1) multiplyIntoImpl(f_, f_incr, cell, qp, w);
}

void transformWeightedGradientBF (
  const Field<2>& F, const RealType& F_det,
  const PHX::MDField<RealType const, Cell, Node, QuadPoint, Dim>& w_grad_bf,
  const int cell, const int pt, const int node, RealType w[3])
{
  const int nd = w_grad_bf.dimension(3);
  for (int k = 0; k < nd; ++k) {
    w[k] = 0;
    for (int i = 0; i < nd; ++i)
      w[k] += (w_grad_bf(cell, node, pt, i) * F()(cell, pt, i, k));
    w[k] /= F_det;
  }
}

#undef loopf
#undef loop

aadapt_rc_eti_class(Field)
#define eti(ad_type, rank)                                              \
  template void Field<rank>::addTo<ad_type>(Tensor<ad_type, rank>::type&) const;
aadapt_rc_apply_to_all_ad_types_all_ranks(eti)
#undef eti
#define eti(ad_type, rank)                                              \
  template void Field<rank>::addTo<ad_type>(                            \
    Tensor<ad_type, rank>::type&, const std::size_t, const std::size_t) const;
aadapt_rc_apply_to_all_ad_types_all_ranks(eti)
#undef eti
#define eti(ad_type, arg2)                                              \
  template void Field<2>::multiplyInto<ad_type>(Tensor<ad_type,2>::type&) const;
aadapt_rc_apply_to_all_ad_types(eti,)
#undef eti
#define eti(ad_type, arg2)                                              \
  template void Field<2>::multiplyInto<ad_type>(                        \
    Tensor<ad_type, 2>::type&, const std::size_t, const std::size_t) const;
aadapt_rc_apply_to_all_ad_types(eti,)
#undef eti

} // namespace rc
} // namespace AAdapt
