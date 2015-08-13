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
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
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
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
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
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
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
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
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
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::MPResidual, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPResidual::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};

//
// MPJacobian
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::MPJacobian, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPJacobian::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};

//
// MPTangent
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::MPTangent, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPTangent::ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};
#endif

} // namespace LCM

#include "MiniLinearSolver_Def.hpp"

#endif // LCM_MiniLinearSolver_hpp
