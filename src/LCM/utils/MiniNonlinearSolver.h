//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniNonlinearSolver_h)
#define LCM_MiniNonlinearSolver_h

#include <type_traits>

#include "PHAL_AlbanyTraits.hpp"
#include "Intrepid_MiniTensor_Solvers.h"

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
template<typename S, typename T, int N>
struct peel
{
  // This ugly return type is to avoid matching Tensor types.
  // If it does not match then it just becomes T.
  typename Intrepid::disable_if_c<Intrepid::order_1234<S>::value, T>::type
  operator()(S const & s)
  {
    T const
    t = s;

    return t;
  }
};

namespace {

using RT = PHAL::AlbanyTraits::Residual::ScalarT;
using JT = PHAL::AlbanyTraits::Jacobian::ScalarT;

template<int N>
using AD = Intrepid::FAD<RealType, N>;

} // anonymous namespace

template<int N>
struct peel<RT, RealType, N>
{
  RealType
  operator()(RT const & s)
  {
    RealType const
    t = s;

    return t;
  }
};

template<int N>
struct peel<JT, RealType, N>
{
  RealType
  operator()(JT const & s)
  {
    RealType const
    t = Sacado::Value<JT>::eval(s);

    return t;
  }
};

template<int N>
struct peel<RT, AD<N>, N>
{
  RealType
  operator()(RT const & s)
  {
    RealType const
    t = s;

    return t;
  }
};

template<int N>
struct peel<JT, AD<N>, N>
{
  RealType
  operator()(JT const & s)
  {
    RealType const
    t = Sacado::Value<JT>::eval(s);

    return t;
  }
};

template<typename S, typename T, int N>
struct peel<Intrepid::Vector<S, N>, Intrepid::Vector<T, N>, N>
{
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
      t[i] = peel<S, T, N>()(s[i]);
    }

    return t;
  }
};

template<typename S, typename T, int N>
struct peel<Intrepid::Tensor<S, N>, Intrepid::Tensor<T, N>, N>
{
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
      t[i] = peel<S, T, N>()(s[i]);
    }

    return t;
  }
};

template<typename S, typename T, int N>
struct peel<Intrepid::Tensor3<S, N>, Intrepid::Tensor3<T, N>, N>
{
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
      t[i] = peel<S, T, N>()(s[i]);
    }

    return t;
  }
};

template<typename S, typename T, int N>
struct peel<Intrepid::Tensor4<S, N>, Intrepid::Tensor4<T, N>, N>
{
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
      t[i] = peel<S, T, N>()(s[i]);
    }

    return t;
  }
};

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // LCM_MiniNonlinearSolver_h
