//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_h)
#define Intrepid_NonlinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include "Intrepid_MiniTensor_Solvers.h"

namespace LCM{

///
/// MiniNonlinear Solver Base class
/// NLS: Nonlinear System
///
template<typename EvalT, typename NLS,
Intrepid::Index N = Intrepid::DYNAMIC>
class MiniNonlinearSolver_Base
{
public:
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using NLM = typename Intrepid::NonlinearMethod_Base<NLS, ValueT, N>;

  MiniNonlinearSolver_Base(NLM & nlm) : nonlinear_method_(nlm)
  {}

  virtual
  ~MiniNonlinearSolver_Base()
  {}

  virtual
  void
  solve(NLS const & nls, Intrepid::Vector<ScalarT, N> & x)
  {}

  virtual
  NLM &
  getNonlinearMethod() final
  {
    return nonlinear_method_;
  }

protected:
  NLM &
  nonlinear_method_;
};

//
// Specializations
//
template<typename EvalT, typename NLS,
Intrepid::Index N = Intrepid::DYNAMIC>
class MiniNonlinearSolver;

//
// Residual
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, NLS, N>
{
public:
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using NLM = typename Intrepid::NonlinearMethod_Base<NLS, ValueT, N>;

  MiniNonlinearSolver(NLM & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, NLS, N>(nlm)
  {}

  void
  solve(NLS const & nls, Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Jacobian
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Jacobian, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Jacobian, NLS, N>
{
public:
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using NLM = typename Intrepid::NonlinearMethod_Base<NLS, ValueT, N>;

  MiniNonlinearSolver(NLM & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Jacobian, NLS, N>(nlm)
  {}

  void
  solve(NLS const & nls, Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Tangent
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Tangent, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Tangent, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using NLM = typename Intrepid::NonlinearMethod_Base<NLS, ValueT, N>;

  MiniNonlinearSolver(NLM & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Tangent, NLS, N>(nlm)
  {}

  void
  solve(NLS const & nls, Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Distributed Parameter Derivative
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::DistParamDeriv, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv,
    NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using NLM = typename Intrepid::NonlinearMethod_Base<NLS, ValueT, N>;

  MiniNonlinearSolver(NLM & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, NLS, N>(nlm)
  {}

  void
  solve(NLS const & nls, Intrepid::Vector<ScalarT, N> & x) override;
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::SGResidual, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::SGResidual, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGResidual::ScalarT;
};

//
// SGJacobian
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::SGJacobian, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::SGJacobian, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGJacobian::ScalarT;
};

//
// SGTangent
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::SGTangent, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::SGTangent, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGTangent::ScalarT;
};
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::MPResidual, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::MPResidual, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPResidual::ScalarT;
};

//
// MPJacobian
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::MPJacobian, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::MPJacobian, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPJacobian::ScalarT;
};

//
// MPTangent
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::MPTangent, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::MPTangent, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPTangent::ScalarT;
};
#endif

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
