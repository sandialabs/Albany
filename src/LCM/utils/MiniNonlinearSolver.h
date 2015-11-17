//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniNonlinearSolver_h)
#define LCM_MiniNonlinearSolver_h

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
  T
  operator()(S const & s)
  {
    T const
    t = s;

    return t;
  }
};

template<int N>
struct peel
<PHAL::AlbanyTraits::Residual::ScalarT, RealType, N>
{
  RealType
  operator()(PHAL::AlbanyTraits::Residual::ScalarT const & s)
  {
    RealType const
    t = s;

    return t;
  }
};

template<int N>
struct peel
<PHAL::AlbanyTraits::Jacobian::ScalarT, RealType, N>
{
  RealType
  operator()(PHAL::AlbanyTraits::Jacobian::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<PHAL::AlbanyTraits::Jacobian::ScalarT>::eval(s);

    return t;
  }
};

template<int N>
struct peel
<PHAL::AlbanyTraits::Residual::ScalarT, Intrepid::FAD<RealType, N>, N>
{
  RealType
  operator()(PHAL::AlbanyTraits::Residual::ScalarT const & s)
  {
    RealType const
    t = s;

    return t;
  }
};

template<int N>
struct peel
<PHAL::AlbanyTraits::Jacobian::ScalarT, Intrepid::FAD<RealType, N>, N>
{
  RealType
  operator()(PHAL::AlbanyTraits::Jacobian::ScalarT const & s)
  {
    RealType const
    t = Sacado::Value<PHAL::AlbanyTraits::Jacobian::ScalarT>::eval(s);

    return t;
  }
};

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // LCM_MiniNonlinearSolver_h
