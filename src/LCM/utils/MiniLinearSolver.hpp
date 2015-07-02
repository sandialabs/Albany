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
template<typename EvalT, typename Traits>
class MiniLinearSolver_Base
{
public:
  typedef typename EvalT::ScalarT ScalarT;

  MiniLinearSolver_Base();

  Teuchos::LAPACK<int, RealType> lapack;

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Specializations
//

template<typename EvalT, typename Traits> class MiniLinearSolver;

//
// Residual
//
template<typename Traits>
class MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Residual, Traits>
{
public:

  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Jacobian
//
template<typename Traits>
class MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
public:

  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Tangent
//
template<typename Traits>
class MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
public:

  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};

//
// Distribured Parameter Derivative
//
template<typename Traits>
class MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits> :
    public MiniLinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
public:

  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

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

  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

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

  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

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

  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

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

  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

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

  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

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
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;

  MiniLinearSolver();

  void solve(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT> const & A,
      Intrepid::Vector<ScalarT> const & b,
      Intrepid::Vector<ScalarT> & x);
};
#endif // ALBANY_ENSEMBLE

} // namespace LCM

#include "MiniLinearSolver_Def.hpp"

#endif // LCM_MiniLinearSolver_hpp
