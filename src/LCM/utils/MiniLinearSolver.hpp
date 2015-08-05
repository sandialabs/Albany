//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniLinearSolver_hpp)
#define LCM_MiniLinearSolver_hpp

#include "PHAL_AlbanyTraits.hpp"
#include <Intrepid_MiniTensor.h>

namespace LCM
{

///
/// Mini Linear Solver Base class
///
template<typename EvalT, typename Traits, Intrepid::Index N = Intrepid::DYNAMIC>
class MiniLinearSolver_Base
{
public:
  using ScalarT = typename EvalT::ScalarT;

  MiniLinearSolver_Base();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Specializations
//

template<typename EvalT, typename Traits, Intrepid::Index N = Intrepid::DYNAMIC>
class MiniLinearSolver;

//
// Residual
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Residual, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Residual::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Jacobian
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Jacobian::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Tangent
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Tangent, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Tangent::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Distribured Parameter Derivative
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::DistParamDeriv::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Stochastic Galerkin Residual
//
#ifdef ALBANY_SG
template<typename Traits>
class MiniLinearSolver< PHAL::AlbanyTraits::SGResidual, Traits> :
public MiniLinearSolver_Base< PHAL::AlbanyTraits::SGResidual, Traits>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGResidual::ScalarT;

  MiniLinearSolver();

  template <Intrepid::Index N>
  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Stochastic Galerkin Jacobian
//
template<typename Traits>
class MiniLinearSolver< PHAL::AlbanyTraits::SGJacobian, Traits> :
public MiniLinearSolver_Base< PHAL::AlbanyTraits::SGJacobian, Traits>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGJacobian::ScalarT;

  MiniLinearSolver();

  template <Intrepid::Index N>
  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Stochastic Galerkin Tangent
//
template<typename Traits>
class MiniLinearSolver< PHAL::AlbanyTraits::SGTangent, Traits> :
public MiniLinearSolver_Base< PHAL::AlbanyTraits::SGTangent, Traits>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGTangent::ScalarT;

  MiniLinearSolver();

  template <Intrepid::Index N>
  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE 
//
// Multi-Point Residual
//
template<typename Traits>
class MiniLinearSolver< PHAL::AlbanyTraits::MPResidual, Traits> :
public MiniLinearSolver_Base< PHAL::AlbanyTraits::MPResidual, Traits>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPResidual::ScalarT;

  MiniLinearSolver();

  template <Intrepid::Index N>
  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Multi-Point Jacobian
//
template <typename Traits>
class MiniLinearSolver< PHAL::AlbanyTraits::MPJacobian, Traits> :
public MiniLinearSolver_Base< PHAL::AlbanyTraits::MPJacobian, Traits>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPJacobian::ScalarT;

  MiniLinearSolver();

  template <Intrepid::Index N>
  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Multi-Point Tangent
//
template<typename Traits>
class MiniLinearSolver< PHAL::AlbanyTraits::MPTangent, Traits> :
public MiniLinearSolver_Base< PHAL::AlbanyTraits::MPTangent, Traits>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPTangent::ScalarT;

  MiniLinearSolver();

  template <Intrepid::Index N>
  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};
#endif // ALBANY_ENSEMBLE

} // namespace LCM

#include "MiniLinearSolver_Def.hpp"

#endif // LCM_MiniLinearSolver_hpp
