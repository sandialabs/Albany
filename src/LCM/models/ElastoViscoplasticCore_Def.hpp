//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>

//
// Define nonlinear system for EV with a Gurson failure surface
//
template <typename EvalT, minitensor::Index M>
EV::ElastoViscoplasticNLS<EvalT, M>::ElastoViscoplasticNLS(
    RealType                              dt,
    RealType                              kw,
    RealType                              en,
    RealType                              sn,
    RealType                              fn,
    RealType                              eHn,
    RealType                              eHn_coeff,
    RealType                              sHn,
    RealType                              fHen,
    RealType                              fHen_coeff,
    RealType                              alpha1,
    RealType                              alpha2,
    RealType                              Ra,
    RealType                              q1,
    RealType                              q2,
    RealType                              q3,
    RealType                              fcrit,
    RealType                              ffail,
    S                                     mu,
    S                                     bulk,
    S                                     Y,
    S                                     H,
    S                                     Rd,
    S                                     rate_coeff,
    S                                     rate_exp,
    S                                     p,
    minitensor::Tensor<S, MAX_DIM> const& s)
    : dt_(dt),
      kw_(kw),
      en_(en),
      sn_(sn),
      fn_(fn),
      eHn_(eHn),
      eHn_coeff_(eHn_coeff),
      sHn_(sHn),
      fHen_(fHen),
      fHen_coeff_(fHen_coeff),
      alpha1_(alpha1),
      alpha2_(alpha2),
      Ra_(Ra),
      q1_(q1),
      q2_(q2),
      q3_(q3),
      fcrit_(fcrit),
      ffail_(ffail),
      mu_(mu),
      bulk_(bulk),
      Y_(Y),
      H_(H),
      Rd_(Rd),
      rate_coeff_(rate_coeff),
      rate_exp_(rate_exp),
      p_(p),
      s_(s)
{
  num_dim_ = s_.get_dimension();
}

template <typename EvalT, minitensor::Index M>
template <typename T, minitensor::Index N>
T
EV::ElastoViscoplasticNLS<EvalT, M>::value(minitensor::Vector<T, N> const& x)
{
  return Base::value(*this, x);
}

template <typename EvalT, minitensor::Index M>
template <typename T, minitensor::Index N>
minitensor::Vector<T, N>
EV::ElastoViscoplasticNLS<EvalT, M>::gradient(
    minitensor::Vector<T, N> const& x) const
{
  auto const num_unknowns = x.get_dimension();

  minitensor::Vector<T, N> residual(num_unknowns);

  return residual;
}

// Nonlinear system, residual based on slip increments
template <typename EvalT, minitensor::Index M>
template <typename T, minitensor::Index N>
minitensor::Tensor<T, N>
EV::ElastoViscoplasticNLS<EvalT, M>::hessian(minitensor::Vector<T, N> const& x)
{
  return Base::hessian(*this, x);
}
