//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_h)
#define Intrepid_NonlinearSolver_h

#include <memory>

#include "PHAL_AlbanyTraits.hpp"
#include "MiniUtils.h"

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

  MiniNonlinearSolver_Base(NonlinearMethod_Base<NLS, ValueT, N> & nlm) :
    nonlinear_method_(nlm)
  {
  }

  virtual
  ~MiniNonlinearSolver_Base() {}

  virtual
  void
  solve(NLS & nls, Intrepid::Vector<ScalarT, N> & x) {}

  virtual
  NonlinearMethod_Base<NLS, ValueT, N> &
  getNonlinearMethod() final
  {
    return nonlinear_method_;
  }

protected:
  NonlinearMethod_Base<NLS, ValueT, N> &
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

  MiniNonlinearSolver(NonlinearMethod_Base<NLS, ValueT, N> & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, NLS, N>(nlm)
  {
  }

  void
  solve(NLS & nls, Intrepid::Vector<ScalarT, N> & x) override;
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

  MiniNonlinearSolver(NonlinearMethod_Base<NLS, ValueT, N> & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Jacobian, NLS, N>(nlm)
  {
  }

  void
  solve(NLS & nls, Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Tangent
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Tangent, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Tangent, NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;
};

//
// Distribured Parameter Derivative
//
template<typename NLS, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::DistParamDeriv, NLS, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv,
    NLS, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;
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
