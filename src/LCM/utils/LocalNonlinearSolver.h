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
  template<typename GlobalT, typename LocalT> 
  class LocalNonlinearSolver_Base
  {
  public:
    typedef typename GlobalT::ScalarT ScalarT;
    LocalNonlinearSolver_Base();
    ~LocalNonlinearSolver_Base() {};
    Teuchos::LAPACK<int,RealType> lapack;
    virtual void solve(std::vector<LocalT> & F, std::vector<LocalT> & X) = 0;
    virtual void computeFadInfo(std::vector<LocalT> & localF, 
                                std::vector<LocalT> & localX, 
                                std::vector<ScalarT> & globalF, 
                                std::vector<ScalarT> & globalX) = 0;
  };

  // ---------------------------------------------------------------------
  // Specializations
  // ---------------------------------------------------------------------

  template<typename GlobalT, typename LocalT> class LocalNonlinearSolver;

  // ---------------------------------------------------------------------
  // Residual
  // ---------------------------------------------------------------------
  template<>
  class LocalNonlinearSolver<PHAL::AlbanyTraits::Residual, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base<PHAL::AlbanyTraits::Residual, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Jacobian
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::Jacobian, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::Jacobian, Sacado::Fad::DFad<RealType> > 
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Tangent
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::Tangent, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::Tangent, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Residual
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::SGResidual, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGResidual, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Jacobian
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::SGJacobian, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGJacobian, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Tangent
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::SGTangent, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::SGTangent, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Multi-Point Residual
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::MPResidual, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPResidual, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Multi-Point Jacobian
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::MPJacobian, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPJacobian, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };

  // ---------------------------------------------------------------------
  // Multi-Point Tangent
  // ---------------------------------------------------------------------
  template <>
  class LocalNonlinearSolver< PHAL::AlbanyTraits::MPTangent, Sacado::Fad::DFad<RealType> > :
    public LocalNonlinearSolver_Base< PHAL::AlbanyTraits::MPTangent, Sacado::Fad::DFad<RealType> >
  {
  public:
    void solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X);
    void computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                        std::vector<Sacado::Fad::DFad<RealType> > & localX,
                        std::vector<ScalarT> & globalF, 
                        std::vector<ScalarT> & globalX);
  };
}

#include "LocalNonlinearSolver.t.cc"

#endif //LCM_LocalNonlienarSolver.h
