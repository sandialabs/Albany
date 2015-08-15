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
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  virtual
  ~NewtonSolver_Base() {}

  virtual
  void
  solve(Residual const & residual, Intrepid::Vector<ScalarT, N> & x) {}

public:
  Intrepid::Index
  maximum_number_iterations{16};

  ValueT
  relative_tolerance{1.0e-10};

  ValueT
  absolute_tolerance{1.0e-10};
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
public:
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  void
  solve(
      Residual const & residual,
      Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Jacobian
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Jacobian, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Jacobian, Residual, N>
{
public:
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};

//
// Tangent
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::Tangent, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::Tangent, Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};

//
// Distribured Parameter Derivative
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::DistParamDeriv, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::DistParamDeriv,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
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
  using ScalarT = typename PHAL::AlbanyTraits::SGResidual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};

//
// SGJacobian
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGJacobian, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGJacobian,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGJacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};

//
// SGTangent
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::SGTangent, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::SGTangent,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGTangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
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
  using ScalarT = typename PHAL::AlbanyTraits::MPResidual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};

//
// MPJacobian
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPJacobian, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPJacobian,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPJacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};

//
// MPTangent
//
template<typename Residual, Intrepid::Index N>
class NewtonSolver<PHAL::AlbanyTraits::MPTangent, Residual, N> :
    public NewtonSolver_Base<PHAL::AlbanyTraits::MPTangent,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPTangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;
};
#endif

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
