//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_LocalNonlinearSolver_hpp)
#define LCM_LocalNonlinearSolver_hpp

#include "PHAL_AlbanyTraits.hpp"
#include <Teuchos_LAPACK.hpp>
#include <Sacado.hpp>

namespace LCM
{

///
/// Local Nonlinear Solver Base class
///
template<typename EvalT, typename Traits>
class LocalNonlinearSolver_Base
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  LocalNonlinearSolver_Base();
  ~LocalNonlinearSolver_Base()
  {
  }
  ;
  Teuchos::LAPACK<int, RealType> lapack;
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Specializations
// -----------------------------------------------------------------------------

template<typename EvalT, typename Traits> class LocalNonlinearSolver;

// -----------------------------------------------------------------------------
// Residual
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::Residual, Traits> :
    public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Jacobian
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::Jacobian, Traits> :
    public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Tangent
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::Tangent, Traits> :
    public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Distribured Parameter Derivative
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits> :
    public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Stochastic Galerkin Residual
// -----------------------------------------------------------------------------
#ifdef ALBANY_SG_MP
template<typename Traits>
class LocalNonlinearSolver< PHAL::AlbanyTraits::SGResidual, Traits> :
public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGResidual, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Stochastic Galerkin Jacobian
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver< PHAL::AlbanyTraits::SGJacobian, Traits> :
public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGJacobian, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Stochastic Galerkin Tangent
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver< PHAL::AlbanyTraits::SGTangent, Traits> :
public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGTangent, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Multi-Point Residual
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver< PHAL::AlbanyTraits::MPResidual, Traits> :
public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPResidual, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Multi-Point Jacobian
// -----------------------------------------------------------------------------
template <typename Traits>
class LocalNonlinearSolver< PHAL::AlbanyTraits::MPJacobian, Traits> :
public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPJacobian, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};

// -----------------------------------------------------------------------------
// Multi-Point Tangent
// -----------------------------------------------------------------------------
template<typename Traits>
class LocalNonlinearSolver< PHAL::AlbanyTraits::MPTangent, Traits> :
public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPTangent, Traits>
{
public:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  LocalNonlinearSolver();
  void solve(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
  void computeFadInfo(std::vector<ScalarT> & A,
      std::vector<ScalarT> & X,
      std::vector<ScalarT> & B);
};
#endif //ALBANY_SG_MP
}

#include "LocalNonlinearSolver_Def.hpp"

#endif //LCM_LocalNonlienarSolver.h
