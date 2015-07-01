//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

template<typename EvalT, typename Traits>
MiniLinearSolver_Base<EvalT, Traits>::MiniLinearSolver_Base() :
    lapack()
{
}

// -----------------------------------------------------------------------------
// Specializations
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Residual
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::Residual, Traits>()
{
}

template<typename Traits>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::
solve(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // system size
  int numLocalVars = B.size();

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // call LAPACK
  this->lapack.GESV(numLocalVars, 1, &A[0], numLocalVars, &IPIV[0], &B[0],
      numLocalVars, &info);

  // increment the solution
  for (int i(0); i < numLocalVars; ++i)
    X[i] -= B[i];
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::
computeFadInfo(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // no-op
}

// -----------------------------------------------------------------------------
// Jacobian
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits>()
{
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::
solve(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // system size
  int numLocalVars = B.size();

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // fill B and dBdX
  std::vector<RealType> F(numLocalVars);
  std::vector<RealType> dFdX(numLocalVars * numLocalVars);
  for (int i(0); i < numLocalVars; ++i)
      {
    F[i] = B[i].val();
    for (int j(0); j < numLocalVars; ++j)
        {
      dFdX[i + numLocalVars * j] = A[i + numLocalVars * j].val();
    }
  }

  // call LAPACK
  this->lapack.GESV(numLocalVars, 1, &dFdX[0], numLocalVars, &IPIV[0], &F[0],
      numLocalVars, &info);

  // increment the solution
  for (int i(0); i < numLocalVars; ++i)
    X[i].val() -= F[i];

}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::
computeFadInfo(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // local system size
  int numLocalVars = B.size();
  int numGlobalVars = B[0].size();
  TEUCHOS_TEST_FOR_EXCEPTION(numGlobalVars == 0, std::logic_error,
      "In MiniLinearSolver<Jacobian> the numGLobalVars is zero where it should be positive\n");

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // extract sensitivities of objective function(s) wrt p
  std::vector<RealType> dBdP(numLocalVars * numGlobalVars);
  for (int i(0); i < numLocalVars; ++i) {
    for (int j(0); j < numGlobalVars; ++j) {
      dBdP[i + numLocalVars * j] = B[i].dx(j);
    }
  }

  // extract the jacobian
  std::vector<RealType> dBdX(A.size());
  for (int i(0); i < numLocalVars; ++i) {
    for (int j(0); j < numLocalVars; ++j) {
      dBdX[i + numLocalVars * j] = A[i + numLocalVars * j].val();
    }
  }
  // call LAPACK to simultaneously solve for all dXdP
  this->lapack.GESV(numLocalVars, numGlobalVars, &dBdX[0], numLocalVars,
      &IPIV[0], &dBdP[0], numLocalVars, &info);

  // unpack into globalX (recall that LAPACK stores dXdP in dBdP)
  for (int i(0); i < numLocalVars; ++i)
      {
    X[i].resize(numGlobalVars);
    for (int j(0); j < numGlobalVars; ++j)
        {
      X[i].fastAccessDx(j) = -dBdP[i + numLocalVars * j];
    }
  }
}

// -----------------------------------------------------------------------------
// Tangent
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::Tangent, Traits>()
{
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::
solve(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // system size
  int numLocalVars = B.size();

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // fill B and dBdX
  std::vector<RealType> F(numLocalVars);
  std::vector<RealType> dFdX(numLocalVars * numLocalVars);
  for (int i(0); i < numLocalVars; ++i)
      {
    F[i] = B[i].val();
    for (int j(0); j < numLocalVars; ++j)
        {
      dFdX[i + numLocalVars * j] = A[i + numLocalVars * j].val();
    }
  }

  // call LAPACK
  this->lapack.GESV(numLocalVars, 1, &dFdX[0], numLocalVars, &IPIV[0], &F[0],
      numLocalVars, &info);

  // increment the solution
  for (int i(0); i < numLocalVars; ++i)
    X[i].val() -= F[i];
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::
computeFadInfo(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // local system size
  int numLocalVars = B.size();
  int numGlobalVars = B[0].size();
  TEUCHOS_TEST_FOR_EXCEPTION(numGlobalVars == 0, std::logic_error,
      "In MiniLinearSolver<Tangent, Traits> the numGLobalVars is zero where it should be positive\n");

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // extract sensitivites of objective function(s) wrt p
  std::vector<RealType> dBdP(numLocalVars * numGlobalVars);
  for (int i(0); i < numLocalVars; ++i) {
    for (int j(0); j < numGlobalVars; ++j) {
      dBdP[i + numLocalVars * j] = B[i].dx(j);
    }
  }

  // extract the jacobian
  std::vector<RealType> dBdX(A.size());
  for (int i(0); i < numLocalVars; ++i) {
    for (int j(0); j < numLocalVars; ++j) {
      dBdX[i + numLocalVars * j] = A[i + numLocalVars * j].val();
    }
  }

  // call LAPACK to simultaneously solve for all dXdP
  this->lapack.GESV(numLocalVars, numGlobalVars, &dBdX[0], numLocalVars,
      &IPIV[0], &dBdP[0], numLocalVars, &info);

  // unpack into globalX (recall that LAPACK stores dXdP in dBdP)
  for (int i(0); i < numLocalVars; ++i)
      {
    X[i].resize(numGlobalVars);
    for (int j(0); j < numGlobalVars; ++j)
        {
      X[i].fastAccessDx(j) = -dBdP[i + numLocalVars * j];
    }
  }
}

// -----------------------------------------------------------------------------
// DistParamDeriv
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>()
{
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
solve(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // system size
  int numLocalVars = B.size();

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // fill B and dBdX
  std::vector<RealType> F(numLocalVars);
  std::vector<RealType> dFdX(numLocalVars * numLocalVars);
  for (int i(0); i < numLocalVars; ++i)
      {
    F[i] = B[i].val();
    for (int j(0); j < numLocalVars; ++j)
        {
      dFdX[i + numLocalVars * j] = A[i + numLocalVars * j].val();
    }
  }

  // call LAPACK
  this->lapack.GESV(numLocalVars, 1, &dFdX[0], numLocalVars, &IPIV[0], &F[0],
      numLocalVars, &info);

  // increment the solution
  for (int i(0); i < numLocalVars; ++i)
    X[i].val() -= F[i];
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
computeFadInfo(
    std::vector<ScalarT> & A,
    std::vector<ScalarT> & X,
    std::vector<ScalarT> & B)
{
  // local system size
  int numLocalVars = B.size();
  int numGlobalVars = B[0].size();
  TEUCHOS_TEST_FOR_EXCEPTION(numGlobalVars == 0, std::logic_error,
      "In MiniLinearSolver<Tangent, Traits> the numGLobalVars is zero where it should be positive\n");

  // data for the LAPACK call below
  int info(0);
  std::vector<int> IPIV(numLocalVars);

  // extract sensitivites of objective function(s) wrt p
  std::vector<RealType> dBdP(numLocalVars * numGlobalVars);
  for (int i(0); i < numLocalVars; ++i) {
    for (int j(0); j < numGlobalVars; ++j) {
      dBdP[i + numLocalVars * j] = B[i].dx(j);
    }
  }

  // extract the jacobian
  std::vector<RealType> dBdX(A.size());
  for (int i(0); i < numLocalVars; ++i) {
    for (int j(0); j < numLocalVars; ++j) {
      dBdX[i + numLocalVars * j] = A[i + numLocalVars * j].val();
    }
  }

  // call LAPACK to simultaneously solve for all dXdP
  this->lapack.GESV(numLocalVars, numGlobalVars, &dBdX[0], numLocalVars,
      &IPIV[0], &dBdP[0], numLocalVars, &info);

  // unpack into globalX (recall that LAPACK stores dXdP in dBdP)
  for (int i(0); i < numLocalVars; ++i)
      {
    X[i].resize(numGlobalVars);
    for (int j(0); j < numGlobalVars; ++j)
        {
      X[i].fastAccessDx(j) = -dBdP[i + numLocalVars * j];
    }
  }
}

// -----------------------------------------------------------------------------
// Stochastic Galerkin Residual
// -----------------------------------------------------------------------------
#ifdef ALBANY_SG
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::SGResidual, Traits>()
{}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits>::
solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Stochastic Galerkin types yet\n");
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits>::
computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Stochastic Galerkin types yet\n");
}

// ---------------------------------------------------------------------
// Stochastic Galerkin Jacobian
// ---------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::SGJacobian, Traits>()
{}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits>::
solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Stochastic Galerkin types yet\n");
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits>::
computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Stochastic Galerkin types yet\n");
}

// -----------------------------------------------------------------------------
// Stochastic Galerkin Tangent
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::SGTangent, Traits>()
{}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits>::
solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Stochastic Galerkin types yet\n");
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits>::
computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Stochastic Galerkin types yet\n");
}
#endif 
#ifdef ALBANY_ENSEMBLE 

// -----------------------------------------------------------------------------
// Multi-Point Residual
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::MPResidual, Traits>()
{}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits>::
solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Multi-Point types yet\n");
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits>::
computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Multi-Point types yet\n");
}

// -----------------------------------------------------------------------------
// Multi-Point Jacobian
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::MPJacobian, Traits>()
{}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits>::
solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Multi-Point types yet\n");
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits>::
computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Multi-Point types yet\n");
}

// -----------------------------------------------------------------------------
// Multi-Point Tangent
// -----------------------------------------------------------------------------
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::MPTangent, Traits>()
{}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits>::
solve(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Multi-Point types yet\n");
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits>::
computeFadInfo(std::vector<ScalarT> & A, std::vector<ScalarT> & X, std::vector<ScalarT> & B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"MiniLinearSolver has not been implemented for Multi-Point types yet\n");
}
#endif
// -----------------------------------------------------------------------------
}

