///
/// \file LocalNonlinearSolver.t.cc
/// A templated nonlinear solver for local (integration point) computations
/// \author Jake Ostien
///

namespace LCM {

  template<typename EvalT>
  LocalNonlinearSolver_Base<EvalT>::LocalNonlinearSolver_Base() :
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
  LocalNonlinearSolver<PHAL::AlbanyTraits::Residual>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    // system size
    int numLocalVars = B.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // call LAPACK
    lapack.GESV(numLocalVars, numLocalVars, &A[0], numLocalVars, &IPIV[0], &B[0], numLocalVars, &info);

    // increment the solution
    for(int i(0); i < numLocalVars; ++i)
      X[i] -= B[i];
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Residual>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    // no-op
  }

  // ---------------------------------------------------------------------
  // Jacobian
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Jacobian>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    // system size
    int numLocalVars = B.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill B and dBdX
    std::vector<RealType> F(numLocalVars);
    std::vector<RealType> dFdX(numLocalVars*numLocalVars);
    for(int i(0); i < numLocalVars; ++i)
    {
      F[i] = B[i].val();
      for(int j(0); j < numLocalVars; ++j)
      {
        dFdX[numLocalVars * i + j] = A[numLocalVars * i + j].val();
      }
    }

    // call LAPACK
    lapack.GESV(numLocalVars, numLocalVars, &dFdX[0], numLocalVars, &IPIV[0], &F[0], numLocalVars, &info);

    // increment the solution
    for(int i(0); i < numLocalVars; ++i)
      X[i].val() -= F[i];
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Jacobian>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    // local system size
    int numLocalVars = B.size();
    int numGlobalVars = B[0].size();
    TEUCHOS_TEST_FOR_EXCEPTION( numGlobalVars == 0, std::logic_error, 
                                "In LocalNonlinearSolver<Jacobian> the numGLobalVars is zero where it should be positive\n");

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // extract sensitivites of objective function(s) wrt p
    std::vector<RealType> dBdP(numLocalVars*numGlobalVars);
    for (int i(0); i < numLocalVars; ++i)
      for (int j(0); j < numGlobalVars; ++j)
        dBdP[numGlobalVars * i + j] = B[i].dx(j);

    // extract the jacobian
    std::vector<RealType> dBdX(A.size());
    for (int i(0); i < numLocalVars; ++i)
      for (int j(0); j < numLocalVars; ++j )
        dBdX[numLocalVars * i + j] = A[numLocalVars * i + j].val();

    // call LAPACK to simultaneously solve for all dXdP
    lapack.GESV(numLocalVars, numGlobalVars, &dBdX[0], numLocalVars, &IPIV[0], &dBdP[0], numLocalVars, &info);

    // unpack into globalX (recall that LAPACK stores dXdP in dBdP)
    for (int i(0); i < numLocalVars; ++i)
    {
      X[i].resize(numGlobalVars);
      for (int j(0); j < numGlobalVars; ++j)
      {
        X[i].fastAccessDx(j) = dBdP[numGlobalVars * i + j];
      }
    }
  }

  // ---------------------------------------------------------------------
  // Tangent
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Tangent>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    // system size
    int numLocalVars = B.size();

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // fill B and dBdX
    std::vector<RealType> F(numLocalVars);
    std::vector<RealType> dFdX(numLocalVars*numLocalVars);
    for(int i(0); i < numLocalVars; ++i)
    {
      F[i] = B[i].val();
      for(int j(0); j < numLocalVars; ++j)
      {
        dFdX[numLocalVars * i + j] = A[numLocalVars * i + j].val();
      }
    }

    // call LAPACK
    lapack.GESV(numLocalVars, numLocalVars, &dFdX[0], numLocalVars, &IPIV[0], &F[0], numLocalVars, &info);

    // increment the solution
    for(int i(0); i < numLocalVars; ++i)
      X[i].val() -= F[i];
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::Tangent>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    // local system size
    int numLocalVars = B.size();
    int numGlobalVars = B[0].size();
    TEUCHOS_TEST_FOR_EXCEPTION( numGlobalVars == 0, std::logic_error, 
                                "In LocalNonlinearSolver<Tangent> the numGLobalVars is zero where it should be positive\n");

    // data for the LAPACK call below
    int info(0);
    std::vector<int> IPIV(numLocalVars);

    // extract sensitivites of objective function(s) wrt p
    std::vector<RealType> dBdP(numLocalVars*numGlobalVars);
    for (int i(0); i < numLocalVars; ++i)
      for (int j(0); j < numGlobalVars; ++j)
        dBdP[numGlobalVars * i + j] = B[i].dx(j);

    // extract the jacobian
    std::vector<RealType> dBdX(A.size());
    for (int i(0); i < numLocalVars; ++i)
      for (int j(0); j < numLocalVars; ++j )
        dBdX[numLocalVars * i + j] = A[numLocalVars * i + j].val();

    // call LAPACK to simultaneously solve for all dXdP
    lapack.GESV(numLocalVars, numGlobalVars, &dBdX[0], numLocalVars, &IPIV[0], &dBdP[0], numLocalVars, &info);

    // unpack into globalX (recall that LAPACK stores dXdP in dBdP)
    for (int i(0); i < numLocalVars; ++i)
    {
      X[i].resize(numGlobalVars);
      for (int j(0); j < numGlobalVars; ++j)
      {
        X[i].fastAccessDx(j) = dBdP[numGlobalVars * i + j];
      }
    }
  }

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Residual
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGResidual>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGResidual>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Jacobian
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGJacobian>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGJacobian>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  // ---------------------------------------------------------------------
  // Stochastic Galerkin Tangent
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGTangent>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::SGTangent>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Stochastic Galerkin types yet\n");
  }

  // ---------------------------------------------------------------------
  // Multi-Point Residual
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPResidual>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPResidual>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  // ---------------------------------------------------------------------
  // Multi-Point Jacobian
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPJacobian>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPJacobian>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  // ---------------------------------------------------------------------
  // Multi-Point Tangent
  // ---------------------------------------------------------------------
  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPTangent>::
  solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

  void
  LocalNonlinearSolver<PHAL::AlbanyTraits::MPTangent>::
  computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"LocalNonlinearSolver has not been implemented for Multi-Point types yet\n");
  }

}

