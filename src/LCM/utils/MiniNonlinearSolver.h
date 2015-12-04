//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniNonlinearSolver_h)
#define LCM_MiniNonlinearSolver_h

#include <type_traits>

#include "Intrepid_MiniTensor_Solvers.h"
#include "PHAL_AlbanyTraits.hpp"

namespace LCM{

///
/// miniMinimize function that wraps the MiniTensor Nonlinear Solvers
/// and deals with Albany traits and AD sensitivities.
///
template<typename MIN, typename STEP, typename FN, Intrepid::Index N>
void
miniMinimize(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid::Vector<RealType, N> & soln);

template<typename MIN, typename STEP, typename FN, typename T, Intrepid::Index N>
void
miniMinimize(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid::Vector<T, N> & soln);

///
/// Deal with derivative information for all the mini solvers.
/// Call this when a converged solution is obtained on a system that is
/// typed on a FAD type.
/// Assuming that T is a FAD type and S is a simple type.
///
template<typename T, typename S, Intrepid::Index N>
void
computeFADInfo(
    Intrepid::Vector<T, N> const & r,
    Intrepid::Tensor<S, N> const & DrDx,
    Intrepid::Vector<T, N> & x);

///
/// Auxiliary functors that peel off derivative information from Albany::Traits
/// types when not needed and keep it when needed. Used to convert types
/// within MiniSolver function class methods.
/// The type for N must be int to work with Sacado.
///
template<typename EvalT, typename T, int N>
struct peel
{
  using S = typename EvalT::ScalarT;

  // This ugly return type is to avoid matching Tensor types.
  // If it does not match then it just becomes T.
  using RET = typename
      Intrepid::disable_if_c<Intrepid::order_1234<T>::value, T>::type;

  RET
  operator()(S const & s)
  {
    T const
    t = s;

    return t;
  }
};

namespace {

using RE = PHAL::AlbanyTraits::Residual;
using JE = PHAL::AlbanyTraits::Jacobian;
using TE = PHAL::AlbanyTraits::Tangent;
using DE = PHAL::AlbanyTraits::DistParamDeriv;

#ifdef ALBANY_SG
using SGRE = PHAL::AlbanyTraits::SGResidual;
using SGJE = PHAL::AlbanyTraits::SGJacobian;
using SGTE = PHAL::AlbanyTraits::SGTangent;
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
using MPRE = PHAL::AlbanyTraits::MPResidual;
using MPJE = PHAL::AlbanyTraits::MPJacobian;
using MPTE = PHAL::AlbanyTraits::MPTangent;
#endif // ALBANY_ENSEMBLE

template<int N>
using AD = Intrepid::FAD<RealType, N>;

} // anonymous namespace

template<int N>
struct peel<RE, RealType, N>
{
  RealType
  operator()(RE::ScalarT const & s)
  {
    RealType const
    t = s;

    return t;
  }
};

template<int N>
struct peel<JE, RealType, N>
{
  RealType
  operator()(JE::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<typename JE::ScalarT>::eval(s);

    return t;
  }
};

template<int N>
struct peel<TE, RealType, N>
{
  RealType
  operator()(TE::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<typename TE::ScalarT>::eval(s);

    return t;
  }
};

template<int N>
struct peel<DE, RealType, N>
{
  RealType
  operator()(DE::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<typename DE::ScalarT>::eval(s);

    return t;
  }
};

template<int N>
struct peel<RE, AD<N>, N>
{
  RealType
  operator()(typename RE::ScalarT const & s)
  {
    RealType const
    t = s;

    return t;
  }
};

template<int N>
struct peel<JE, AD<N>, N>
{
  RealType
  operator()(JE::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<typename JE::ScalarT>::eval(s);

    return t;
  }
};

template<int N>
struct peel<TE, AD<N>, N>
{
  RealType
  operator()(TE::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<typename TE::ScalarT>::eval(s);

    return t;
  }
};

template<int N>
struct peel<DE, AD<N>, N>
{
  RealType
  operator()(DE::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<typename DE::ScalarT>::eval(s);

    return t;
  }
};

#ifdef ALBANY_SG
template<int N>
struct peel<SGRE, RealType, N>
{
  RealType
  operator()(SGRE::ScalarT const &)
  {
    return 0.0;
  }
};

template<int N>
struct peel<SGJE, RealType, N>
{
  RealType
  operator()(SGJE::ScalarT const &)
  {
    return 0.0;
  }
};

template<int N>
struct peel<SGRE, AD<N>, N>
{
  RealType
  operator()(SGRE::ScalarT const &)
  {
    return 0.0;
  }
};

template<int N>
struct peel<SGJE, AD<N>, N>
{
  RealType
  operator()(SGJE::ScalarT const &)
  {
    return 0.0;
  }
};
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
template<int N>
struct peel<MPRE, RealType, N>
{
  RealType
  operator()(MPRE::ScalarT const &)
  {
    return 0.0;
  }
};

template<int N>
struct peel<MPJE, RealType, N>
{
  RealType
  operator()(MPJE::ScalarT const &)
  {
    return 0.0;
  }
};

template<int N>
struct peel<MPRE, AD<N>, N>
{
  RealType
  operator()(MPRE::ScalarT const &)
  {
    return 0.0;
  }
};

template<int N>
struct peel<MPJE, AD<N>, N>
{
  RealType
  operator()(MPJE::ScalarT const &)
  {
    return 0.0;
  }
};
#endif // ALBANY_ENSEMBLE

template<typename EvalT, typename T, int N>
struct peel_vector
{
  using S = typename EvalT::ScalarT;

  Intrepid::Vector<T, N>
  operator()(Intrepid::Vector<S, N> const & s)
  {
    Intrepid::Index const
    dimension = s.get_dimension();

    Intrepid::Vector<T, N>
    t(dimension);

    Intrepid::Index const
    num_components = s.get_number_components();

    for (Intrepid::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, N>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int N>
struct peel_tensor
{
  using S = typename EvalT::ScalarT;

  Intrepid::Tensor<T, N>
  operator()(Intrepid::Tensor<S, N> const & s)
  {
    Intrepid::Index const
    dimension = s.get_dimension();

    Intrepid::Tensor<T, N>
    t(dimension);

    Intrepid::Index const
    num_components = s.get_number_components();

    for (Intrepid::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, N>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int N>
struct peel_tensor3
{
  using S = typename EvalT::ScalarT;

  Intrepid::Tensor3<T, N>
  operator()(Intrepid::Tensor3<S, N> const & s)
  {
    Intrepid::Index const
    dimension = s.get_dimension();

    Intrepid::Tensor3<T, N>
    t(dimension);

    Intrepid::Index const
    num_components = s.get_number_components();

    for (Intrepid::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, N>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int N>
struct peel_tensor4
{
  using S = typename EvalT::ScalarT;

  Intrepid::Tensor4<T, N>
  operator()(Intrepid::Tensor4<S, N> const & s)
  {
    Intrepid::Index const
    dimension = s.get_dimension();

    Intrepid::Tensor4<T, N>
    t(dimension);

    Intrepid::Index const
    num_components = s.get_number_components();

    for (Intrepid::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, N>()(s[i]);
    }

    return t;
  }
};

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // LCM_MiniNonlinearSolver_h
