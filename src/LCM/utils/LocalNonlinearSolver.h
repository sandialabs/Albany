///
/// \file LocalNonlinearSolver.h
/// A templated nonlinear solver for local (integration point) computations
/// \author Jake Ostien
///
#if !defined(LCM_LocalNonlinearSolver_h)
#define LCM_LocalNonlinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_RCP.hpp>
#include <Sacado.hpp>

namespace LCM {

  ///
  /// Local Nonlinear Solver Base class
  ///
  template<typename EvalT> 
  class LocalNonlinearSolver_Base
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    LocalNonlinearSolver_Base();
    ~LocalNonlinearSolver_Base() {};
    Teuchos::LAPACK<int,RealType> lapack;
    virtual void solve(std::vector<ScalarT> & A, 
                       std::vector<ScalarT> & X,
                       std::vector<ScalarT> & B) = 0;
    virtual void computeFadInfo(std::vector<ScalarT> & A, 
                                std::vector<ScalarT> & X, 
                                std::vector<ScalarT> & B) = 0; 
  };

  // ---------------------------------------------------------------------
  // Specializations
  // ---------------------------------------------------------------------

  template<typename EvalT> class LocalNonlinearSolver;

  // ---------------------------------------------------------------------
  // Residual
  // ---------------------------------------------------------------------
  template<>
  class LocalNonlinearSolver<PHAL::AlbanyTraits::Residual> :
    public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Residual>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Jacobian
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::Jacobian> :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::Jacobian> 
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Tangent
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::Tangent> :
      public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::Tangent>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Residual
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::SGResidual> : 
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGResidual>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Jacobian
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::SGJacobian> :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGJacobian>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Tangent
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::SGTangent> :
      public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGTangent>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Multi-Point Residual
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::MPResidual> :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPResidual>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Multi-Point Jacobian
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::MPJacobian> :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPJacobian>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };

  // ---------------------------------------------------------------------
  // Multi-Point Tangent
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::MPTangent> :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPTangent>
  {
  public:
    void solve(std::vector<ScalarT> & A, 
               std::vector<ScalarT> & X,
               std::vector<ScalarT> & B);
    void computeFadInfo(std::vector<ScalarT> & A, 
                        std::vector<ScalarT> & X,
                        std::vector<ScalarT> & B);
  };
}

#include "LocalNonlinearSolver.t.cc"

#endif //LCM_LocalNonlienarSolver.h
