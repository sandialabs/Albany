//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_LocalNonlinearSolver_hpp)
#define LCM_LocalNonlinearSolver_hpp

#include <Sacado.hpp>
#include <Teuchos_LAPACK.hpp>
#include "PHAL_AlbanyTraits.hpp"

namespace LCM {

///
/// Local Nonlinear Solver Base class
///
template <typename EvalT, typename Traits>
class LocalNonlinearSolver_Base
{
 public:
  typedef typename EvalT::ScalarT ScalarT;
  LocalNonlinearSolver_Base();
  ~LocalNonlinearSolver_Base(){};
  Teuchos::LAPACK<int, RealType> lapack;
  void
  solve(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
  void
  computeFadInfo(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
};

// -----------------------------------------------------------------------------
// Specializations
// -----------------------------------------------------------------------------

template <typename EvalT, typename Traits>
class LocalNonlinearSolver;

// -----------------------------------------------------------------------------
// Residual
// -----------------------------------------------------------------------------
template <typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::Residual, Traits>
    : public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  LocalNonlinearSolver();
  void
  solve(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
  void
  computeFadInfo(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
};

// -----------------------------------------------------------------------------
// Jacobian
// -----------------------------------------------------------------------------
template <typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>
    : public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  LocalNonlinearSolver();
  void
  solve(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
  void
  computeFadInfo(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
};

// -----------------------------------------------------------------------------
// Tangent
// -----------------------------------------------------------------------------
template <typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::Tangent, Traits>
    : public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  LocalNonlinearSolver();
  void
  solve(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
  void
  computeFadInfo(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
};

// -----------------------------------------------------------------------------
// Distributed Parameter Derivative
// -----------------------------------------------------------------------------
template <typename Traits>
class LocalNonlinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public LocalNonlinearSolver_Base<
          PHAL::AlbanyTraits::DistParamDeriv,
          Traits>
{
 public:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  LocalNonlinearSolver();
  void
  solve(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
  void
  computeFadInfo(
      std::vector<ScalarT>& A,
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& B);
};

}  // namespace LCM

#include "LocalNonlinearSolver_Def.hpp"

#endif  // LCM_LocalNonlienarSolver.h
