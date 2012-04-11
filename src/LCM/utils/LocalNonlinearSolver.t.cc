///
/// \file LocalNonlinearSolver.t.cc
/// A templated nonlinear solver for local (integration point) computations
/// \author Jake Ostien
///

namespace LCM {

  template<typename GlobalT, typename LocalT>
  LocalNonlinearSolver_Base<GlobalT,LocalT>::LocalNonlinearSolver_Base() :
    lapack()
  {
  }

  // ---------------------------------------------------------------------
  // Specializations
  // ---------------------------------------------------------------------

  // ---------------------------------------------------------------------
  // Residual
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Residual, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > &F, std::vector<Sacado::Fad::DFad<RealType> > &X)
  {
    // system size
    int numLocalVars = F.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill F and dFdX from F
    std::vector<RealType> f;
    std::vector<RealType> dFdX;
    std::vector<Sacado::Fad::DFad<RealType> >::iterator it;
    for(it = F.begin(); it != F.end(); it++)
    {
      for(int n(0); n < numLocalVars; ++n)
      {
        dFdX.push_back(it->dx(n));
      }
      f.push_back(it->val());
    }

    // call LAPACK
    lapack.GESV(numLocalVars, numLocalVars, &dFdX[0], numLocalVars, &IPIV[0], &f[0], numLocalVars, &info);

    // increment the solution
    for(int i(0); i < numLocalVars; ++i)
      X[i].val() -= f[i];
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Residual, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    // system size
    int numLocalVars = localF.size();

    for (int i(0); i < numLocalVars; ++i)
      globalX[0] = localX[0].val();
  }

  // ---------------------------------------------------------------------
  // Jacobian
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Jacobian, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    // system size
    int numLocalVars = F.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill F and dFdX from F
    std::vector<RealType> f;
    std::vector<RealType> dFdX;
    std::vector<Sacado::Fad::DFad<RealType> >::iterator it;
    for(it = F.begin(); it != F.end(); it++)
    {
      for(int n(0); n < numLocalVars; ++n)
      {
        dFdX.push_back(it->dx(n));
      }
      f.push_back(it->val());
    }

    // call LAPACK
    lapack.GESV(numLocalVars, numLocalVars, &dFdX[0], numLocalVars, &IPIV[0], &f[0], numLocalVars, &info);

    // increment the solution
    for(int i(0); i < numLocalVars; ++i)
      X[i].val() -= f[i];

  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Jacobian, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    // local system size
    int numLocalVars = localF.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill solution
    for (int i(0); i < numLocalVars; ++i)
      globalX[i].val() = localX[i].val();

    // number of external Parameters
    int numGlobalVars = globalF[0].size();
    TEUCHOS_TEST_FOR_EXCEPTION( numGlobalVars == 0, std::logic_error, 
                                "In LocalNonlinearSolver<Jacobian,Dfad> the numGLobalVars is zero where it should be positive\n");

    // extract the jacobian
    std::vector<RealType> dFdX;
    std::vector<Sacado::Fad::DFad<RealType> >::iterator it;
    for(it = localF.begin(); it != localF.end(); it++)
      for(int n(0); n < numLocalVars; ++n)
        dFdX.push_back(it->dx(n));

    // fill in external parameter sensitivities
    std::vector<RealType> dFdP(numLocalVars*numGlobalVars);
    for (int i(0); i < numLocalVars; ++i)
      for (int j(0); j < numGlobalVars; ++j)
        dFdP[numGlobalVars * i + j] = globalF[i].dx(j);
    
    // call LAPACK to simultaneously solve for all dXdP
    lapack.GESV(numLocalVars, numGlobalVars, &dFdX[0], numLocalVars, &IPIV[0], &dFdP[0], numLocalVars, &info);

    // unpack into globalX (recall that LAPACK stores dXdP in dFdP)
    for (int i(0); i < numLocalVars; ++i)
    {
      globalX[i].resize(numGlobalVars);
      for (int j(0); j < numGlobalVars; ++j)
      {
        globalX[i].fastAccessDx(j) = dFdP[numGlobalVars * i + j];
      }
    }
  }

  // ---------------------------------------------------------------------
  // Tangent
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Tangent, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    // system size
    int numLocalVars = F.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill F and dFdX from F
    std::vector<RealType> f;
    std::vector<RealType> dFdX;
    std::vector<Sacado::Fad::DFad<RealType> >::iterator it;
    for(it = F.begin(); it != F.end(); it++)
    {
      for(int n(0); n < numLocalVars; ++n)
      {
        dFdX.push_back(it->dx(n));
      }
      f.push_back(it->val());
    }

    // call LAPACK
    lapack.GESV(numLocalVars, numLocalVars, &dFdX[0], numLocalVars, &IPIV[0], &f[0], numLocalVars, &info);

    // increment the solution
    for(int i(0); i < numLocalVars; ++i)
      X[i].val() -= f[i];

  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Tangent, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    // local system size
    int numLocalVars = localF.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill solution
    for (int i(0); i < numLocalVars; ++i)
      globalX[i].val() = localX[i].val();

    // number of external Parameters
    int numGlobalVars = globalF[0].size();
    TEUCHOS_TEST_FOR_EXCEPTION( numGlobalVars == 0, std::logic_error, 
                                "In LocalNonlinearSolver<Jacobian,Dfad> the numGLobalVars is zero where it should be positive\n");

    // extract the jacobian
    std::vector<RealType> dFdX;
    std::vector<Sacado::Fad::DFad<RealType> >::iterator it;
    for(it = localF.begin(); it != localF.end(); it++)
      for(int n(0); n < numLocalVars; ++n)
        dFdX.push_back(it->dx(n));

    // fill in external parameter sensitivities
    std::vector<RealType> dFdP(numLocalVars*numGlobalVars);
    for (int i(0); i < numLocalVars; ++i)
      for (int j(0); j < numGlobalVars; ++j)
        dFdP[numGlobalVars * i + j] = globalF[i].dx(j);
    
    // call LAPACK to simultaneously solve for all dXdP
    lapack.GESV(numLocalVars, numGlobalVars, &dFdX[0], numLocalVars, &IPIV[0], &dFdP[0], numLocalVars, &info);

    // unpack into globalX (recall that LAPACK stores dXdP in dFdP)
    for (int i(0); i < numLocalVars; ++i)
    {
      globalX[i].resize(numGlobalVars);
      for (int j(0); j < numGlobalVars; ++j)
      {
        globalX[i].fastAccessDx(j) = dFdP[numGlobalVars * i + j];
      }
    }
  }

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Residual
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGResidual, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGResidual, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Jacobian
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGJacobian, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGJacobian, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Tangent
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGTangent, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGTangent, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  // ---------------------------------------------------------------------
  // Multi-Point Residual
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPResidual, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPResidual, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  // ---------------------------------------------------------------------
  // Multi-Point Jacobian
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPJacobian, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPJacobian, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  // ---------------------------------------------------------------------
  // Multi-Point Tangent
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPTangent, Sacado::Fad::DFad<RealType> >::
  solve(std::vector<Sacado::Fad::DFad<RealType> > & F, std::vector<Sacado::Fad::DFad<RealType> > & X)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPTangent, Sacado::Fad::DFad<RealType> >::
  computeFadInfo(std::vector<Sacado::Fad::DFad<RealType> > & localF,
                 std::vector<Sacado::Fad::DFad<RealType> > & localX,
                 std::vector<ScalarT> & globalF, 
                 std::vector<ScalarT> & globalX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

}

