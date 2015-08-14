//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniLinearSolver_h)
#define LCM_MiniLinearSolver_h

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

  virtual ~MiniLinearSolver_Base() {}

  virtual void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) {}

  virtual void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) {}
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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x)  override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::SGResidual, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGResidual::ScalarT;

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// SGJacobian
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::SGJacobian, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGJacobian::ScalarT;

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// SGTangent
//
template<typename Traits, Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::SGTangent, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGTangent::ScalarT;

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};
#endif

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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
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

  void solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};
#endif

} // namespace LCM

#include "MiniLinearSolver.t.h"

#endif // LCM_MiniLinearSolver_h
