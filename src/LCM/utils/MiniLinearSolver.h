//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniLinearSolver_h)
#define LCM_MiniLinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include "Intrepid_MiniTensor_Solvers.h"

namespace LCM
{

///
/// Mini Linear Solver Base class
///
template<typename EvalT, Intrepid::Index N = Intrepid::DYNAMIC>
class MiniLinearSolver_Base
{
public:
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  virtual
  ~MiniLinearSolver_Base() {}

  virtual
  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) {}
};

//
// Specializations
//
template<typename EvalT, Intrepid::Index N = Intrepid::DYNAMIC>
class MiniLinearSolver;

//
// Residual
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Residual, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Residual, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Jacobian
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Jacobian, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Jacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x)  override;
};

//
// Tangent
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::Tangent, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Tangent, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Tangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Distributed Parameter Derivative
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::DistParamDeriv::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::SGResidual, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGResidual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// SGJacobian
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::SGJacobian, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGJacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// SGTangent
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::SGTangent, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGTangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::MPResidual, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPResidual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// MPJacobian
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::MPJacobian, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPJacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// MPTangent
//
template<Intrepid::Index N>
class MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, N> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::MPTangent, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPTangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  void
  solve(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x) override;
};
#endif

} // namespace LCM

#include "MiniLinearSolver.t.h"

#endif // LCM_MiniLinearSolver_h
