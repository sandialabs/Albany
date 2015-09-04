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
///
template<typename EvalT, typename Residual,
Intrepid::Index N = Intrepid::DYNAMIC>
class MiniNonlinearSolver_Base
{
public:
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  MiniNonlinearSolver_Base(NonlinearMethod_Base<Residual, ValueT, N> & nlm) :
    nonlinear_method_(nlm)
  {
  }

  virtual
  ~MiniNonlinearSolver_Base() {}

  virtual
  void
  solve(Residual & residual, Intrepid::Vector<ScalarT, N> & x) {}

  virtual
  NonlinearMethod_Base<Residual, ValueT, N> &
  getNonlinearMethod() final
  {
    return nonlinear_method_;
  }

protected:
  NonlinearMethod_Base<Residual, ValueT, N> &
  nonlinear_method_;
};

//
// Specializations
//
template<typename EvalT, typename Residual,
Intrepid::Index N = Intrepid::DYNAMIC>
class MiniNonlinearSolver;

//
// Residual
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, Residual, N>
{
public:
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  MiniNonlinearSolver(NonlinearMethod_Base<Residual, ValueT, N> & nlm) :
    MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, Residual, N>(nlm)
  {
  }

  void
  solve(Residual & residual, Intrepid::Vector<ScalarT, N> & x) override;
};

//
// Jacobian
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Jacobian, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Residual, N>
{
public:
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;
};

//
// Tangent
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::Tangent, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::Tangent, Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;
};

//
// Distribured Parameter Derivative
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;
};

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::SGResidual, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::SGResidual,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGResidual::ScalarT;
};

//
// SGJacobian
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::SGJacobian, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::SGJacobian,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGJacobian::ScalarT;
};

//
// SGTangent
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::SGTangent, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::SGTangent,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::SGTangent::ScalarT;
};
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::MPResidual, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::MPResidual,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPResidual::ScalarT;
};

//
// MPJacobian
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::MPJacobian, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::MPJacobian,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPJacobian::ScalarT;
};

//
// MPTangent
//
template<typename Residual, Intrepid::Index N>
class MiniNonlinearSolver<PHAL::AlbanyTraits::MPTangent, Residual, N> :
    public MiniNonlinearSolver_Base<PHAL::AlbanyTraits::MPTangent,
    Residual, N>
{
  using ScalarT = typename PHAL::AlbanyTraits::MPTangent::ScalarT;
};
#endif

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
