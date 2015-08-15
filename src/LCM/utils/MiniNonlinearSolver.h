//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_h)
#define Intrepid_NonlinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include <Intrepid_MiniTensor.h>

namespace LCM{

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
template<typename EvalT, typename Residual,
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
template<typename EvalT, typename Residual,
Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonSolver;

//
// Residual
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Residual, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Residual, Residual, N>
{
};

//
// Jacobian
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Jacobian, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Jacobian, Residual, N>
{
};

//
// Tangent
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Tangent, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Tangent, Residual, N>
{
};

//
// Distribured Parameter Derivative
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::DistParamDeriv, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::DistParamDeriv,
    Residual, N>
{
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGResidual, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGResidual,
    Residual, N>
{
};

//
// SGJacobian
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGJacobian, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGJacobian,
    Residual, N>
{
};

//
// SGTangent
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGTangent, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGTangent,
    Residual, N>
{
};
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPResidual, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPResidual,
    Residual, N>
{
};

//
// MPJacobian
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPJacobian, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPJacobian,
    Residual, N>
{
};

//
// MPTangent
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPTangent, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPTangent,
    Residual, N>
{
};
#endif

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
