//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_h)
#define Intrepid_NonlinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include <Intrepid_MiniTensor.h>

namespace LCM
{

///
/// Residual interafce for mini nonlinear solver
/// To use the solver framework, derive from this class and perform
/// residual computations in the compute method.
///
template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class Residual_Base
{
public:
  virtual
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) = 0;

  virtual
  ~Residual_Base() {}
};

///
/// Newton Solver Base class
///
template<typename EvalT, typename Traits, Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonSolver_Base
{
public:
  using ScalarT = typename EvalT::ScalarT;

  NewtonSolver_Base();

  template <typename Residual>
  void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);

  Intrepid::Index
  get_maximum_number_iterations() const
  {
    return max_num_iter_;
  }

  RealType
  get_relative_tolerance() const
  {
    return relative_tolerance_;
  }

  RealType
  get_absolute_tolerance() const
  {
    return absolute_tolerance_;
  }

  void
  set_maximum_number_iterations(Intrepid::Index const n)
  {
    max_num_iter_ = n;
  }

  void
  set_relative_tolerance(RealType const eps)
  {
    relative_tolerance_ = eps;
  }

  void
  set_absolute_tolerance(RealType const eps)
  {
    absolute_tolerance_ = eps;
  }

protected:
  Intrepid::Index
  max_num_iter_{16};

  RealType
  relative_tolerance_{1.0e-10};

  RealType
  absolute_tolerance_{1.0e-10};
};

//
// Specializations
//

template<typename EvalT, typename Traits, Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonSolver;

//
// Residual
//
template<typename Traits, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Residual, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Residual, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Residual::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
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
class NewtonSolver<PHAL::AlbanyTraits::Jacobian, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Jacobian::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
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
class NewtonSolver<PHAL::AlbanyTraits::Tangent, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Tangent, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::Tangent::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
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
class NewtonSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::DistParamDeriv::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename Traits, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGResidual, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGResidual, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGResidual::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};

//
// SGJacobian
//
template<typename Traits, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGJacobian, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGJacobian, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGJacobian::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};

//
// SGTangent
//
template<typename Traits, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGTangent, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGTangent, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::SGTangent::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename Traits, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPResidual, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPResidual, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPResidual::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
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
class NewtonSolver<PHAL::AlbanyTraits::MPJacobian, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPJacobian, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPJacobian::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
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
class NewtonSolver<PHAL::AlbanyTraits::MPTangent, Traits, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPTangent, Traits, N>
{
public:
  using ScalarT = PHAL::AlbanyTraits::MPTangent::ScalarT;

  NewtonSolver();

  template <typename Residual>
  void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  void computeFadInfo(
      Intrepid::Tensor<ScalarT, N> const & A,
      Intrepid::Vector<ScalarT, N> const & b,
      Intrepid::Vector<ScalarT, N> & x);
};
#endif

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
