//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniNonlinearSolver_h)
#define LCM_MiniNonlinearSolver_h

#include <type_traits>

#include "Intrepid2_MiniTensor_Solvers.h"
#include "PHAL_AlbanyTraits.hpp"

namespace LCM{

//
// Class for dealing with Albany traits.
//
template<
typename MIN, typename STEP, typename FN, typename EvalT, Intrepid2::Index N>
struct MiniSolver
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<typename EvalT::ScalarT, N> & soln);
};

//
// MiniSolver class specializations for Albany traits.
//

template<typename MIN, typename STEP, typename FN, Intrepid2::Index N>
struct MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::Residual, N>
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<PHAL::AlbanyTraits::Residual::ScalarT, N> & soln);
};

template<typename MIN, typename STEP, typename FN, Intrepid2::Index N>
struct MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::Jacobian, N>
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<PHAL::AlbanyTraits::Jacobian::ScalarT, N> & soln);
};

template<typename MIN, typename STEP, typename FN, Intrepid2::Index N>
struct MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::Tangent, N>
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<PHAL::AlbanyTraits::Tangent::ScalarT, N> & soln);
};

template<typename MIN, typename STEP, typename FN, Intrepid2::Index N>
struct MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::DistParamDeriv, N>
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<PHAL::AlbanyTraits::DistParamDeriv::ScalarT, N> & soln);
};

///
/// Deal with derivative information for all the mini solvers.
/// Call this when a converged solution is obtained on a system that is
/// typed on a FAD type.
/// Assuming that T is a FAD type and S is a simple type.
///
template<typename T, typename S, Intrepid2::Index N>
void
computeFADInfo(
    Intrepid2::Vector<T, N> const & r,
    Intrepid2::Tensor<S, N> const & DrDx,
    Intrepid2::Vector<T, N> & x);

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
      Intrepid2::disable_if_c<Intrepid2::order_1234<T>::value, T>::type;

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
using AD = Intrepid2::FAD<RealType, N>;

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

// M: number of derivatives
// N: vector/tensor dimension
template<typename EvalT, typename T, int M, int N>
struct peel_vector
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Vector<T, N>
  operator()(Intrepid2::Vector<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Vector<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Tensor<T, N>
  operator()(Intrepid2::Tensor<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Tensor<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor3
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Tensor3<T, N>
  operator()(Intrepid2::Tensor3<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Tensor3<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor4
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Tensor4<T, N>
  operator()(Intrepid2::Tensor4<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Tensor4<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // LCM_MiniNonlinearSolver_h
