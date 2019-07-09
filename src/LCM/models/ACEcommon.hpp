//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_ACEcommon_hpp)
#define LCM_ACEcommon_hpp

#include "Albany_Utils.hpp"
#include "MiniNonlinearSolver.h"

namespace LCM {

std::vector<RealType>
vectorFromFile(std::string const& filename);

RealType
interpolateVectors(
    std::vector<RealType> const& xv,
    std::vector<RealType> const& yv,
    RealType const               x);

namespace {

static RealType const SQ23{std::sqrt(2.0 / 3.0)};

}  // anonymous namespace

//
// ACE nonlinear system for ice and permafrost material models (J2)
//
template <typename EvalT, minitensor::Index M = 1>
class ACE_NLS : public minitensor::
                    Function_Base<ACE_NLS<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

 public:
  ACE_NLS(
      RealType sat_mod,
      RealType sat_exp,
      RealType eqps_old,
      S const& K,
      S const& smag,
      S const& mubar,
      S const& Y)
      : sat_mod_(sat_mod),
        sat_exp_(sat_exp),
        eqps_old_(eqps_old),
        K_(K),
        smag_(smag),
        mubar_(mubar),
        Y_(Y)
  {
  }

  constexpr static char const* const NAME{"ACE NLS"};

  using Base =
      minitensor::Function_Base<ACE_NLS<EvalT, M>, typename EvalT::ScalarT, M>;

  // Default value.
  template <typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const& x)
  {
    return Base::value(*this, x);
  }

  // Explicit gradient.
  template <typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const& x)
  {
    // Firewalls.
    minitensor::Index const dimension = x.get_dimension();

    ALBANY_EXPECT(dimension == Base::DIMENSION);

    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const K     = peel<EvalT, T, N>()(K_);
    T const smag  = peel<EvalT, T, N>()(smag_);
    T const mubar = peel<EvalT, T, N>()(mubar_);
    T const Y     = peel<EvalT, T, N>()(Y_);

    // This is the actual computation of the gradient.
    minitensor::Vector<T, N> r(dimension);

    T const& X     = x(0);
    T const  alpha = eqps_old_ + SQ23 * X;
    T const  H     = K * alpha + sat_mod_ * (1.0 - std::exp(-sat_exp_ * alpha));
    T const  R     = smag - (2.0 * mubar * X + SQ23 * (Y + H));

    r(0) = R;

    return r;
  }

  // Default AD hessian.
  template <typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const& x)
  {
    return Base::hessian(*this, x);
  }

  // Constants.
  RealType const sat_mod_{0.0};
  RealType const sat_exp_{0.0};
  RealType const eqps_old_{0.0};

  // Inputs
  S const& K_;
  S const& smag_;
  S const& mubar_;
  S const& Y_;
};

}  // namespace LCM

#endif  // LCM_ACEcommon_hpp
