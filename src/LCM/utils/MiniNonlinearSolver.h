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
template<typename EvalT, typename Traits, typename Residual,
Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonSolver_Base
{
public:
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>;
  using FAD = typename Sacado::Fad::DFad<ValueT>;

  virtual ~NewtonSolver_Base() {}

  virtual void solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x);

  virtual
  Intrepid::Index
  get_maximum_number_iterations() const final
  {
    return max_num_iter_;
  }

  virtual
  ValueT
  get_relative_tolerance() const final
  {
    return relative_tolerance_;
  }

  virtual
  ValueT
  get_absolute_tolerance() const final
  {
    return absolute_tolerance_;
  }

  virtual
  void
  set_maximum_number_iterations(Intrepid::Index const n) final
  {
    max_num_iter_ = n;
  }

  virtual
  void
  set_relative_tolerance(ValueT const eps) final
  {
    relative_tolerance_ = eps;
  }

  virtual
  void
  set_absolute_tolerance(ValueT const eps) final
  {
    absolute_tolerance_ = eps;
  }

protected:
  Intrepid::Index
  max_num_iter_{16};

  ValueT
  relative_tolerance_{1.0e-10};

  ValueT
  absolute_tolerance_{1.0e-10};
};

//
// Specializations
//
template<typename EvalT, typename Traits,
typename Residual, Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonSolver;

//
// Residual
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Residual, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Residual, Traits, Residual, N>
{
};

//
// Jacobian
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Jacobian, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits, Residual, N>
{
};

//
// Tangent
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Tangent, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Tangent, Traits, Residual, N>
{
};

//
// Distribured Parameter Derivative
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::DistParamDeriv,
    Traits, Residual, N>
{
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGResidual, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGResidual,
    Traits, Residual, N>
{
};

//
// SGJacobian
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGJacobian, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGJacobian,
    Traits, Residual, N>
{
};

//
// SGTangent
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGTangent, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGTangent,
    Traits, Residual, N>
{
};
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPResidual, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPResidual,
    Traits, Residual, N>
{
};

//
// MPJacobian
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPJacobian, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPJacobian,
    Traits, Residual, N>
{
};

//
// MPTangent
//
template<typename Traits, typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPTangent, Traits, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPTangent,
    Traits, Residual, N>
{
};
#endif

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
