//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

template<typename EvalT, typename Traits>
MiniLinearSolver_Base<EvalT, Traits>::MiniLinearSolver_Base()
{
  return;
}

//
// Specializations
//

//
// Residual
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::Residual, Traits>()
{
  return;
}

template<typename Traits>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::
solve(
Intrepid::Tensor<ScalarT> const & A,
Intrepid::Vector<ScalarT> const & b,
Intrepid::Vector<ScalarT> & x)
{
  x -= Intrepid::solve(A, b);
  return;
}

template<typename Traits>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::
computeFadInfo(
Intrepid::Tensor<ScalarT> const & A,
Intrepid::Vector<ScalarT> const & b,
Intrepid::Vector<ScalarT> & x)
{
  // no-op
  return;
}

//
// Jacobian
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::Jacobian, Traits>()
{
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::
solve(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<RealType>
  Df(local_dim);

  Intrepid::Tensor<RealType>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<RealType> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() -= Dx(i);
  }

  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  auto const
  local_dim = b.get_dimension();

  auto const
  global_dim = b[0].size();

  assert(global_dim > 0);

  Intrepid::Matrix<RealType>
  DbDp(local_dim, global_dim);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < global_dim; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<RealType>
  DbDx(local_dim);

  // extract the jacobian
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < local_dim; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<RealType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < local_dim; ++i) {
    x(i).resize(global_dim);
    for (auto j = 0; j < global_dim; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

//
// Tangent
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::MiniLinearSolver() :
    MiniLinearSolver_Base<PHAL::AlbanyTraits::Tangent, Traits>()
{
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::
solve(
    Intrepid::Tensor<ScalarT> const & A ,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<RealType>
  Df(local_dim);

  Intrepid::Tensor<RealType>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<RealType> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() -= Dx(i);
  }

  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  auto const
  local_dim = b.get_dimension();

  auto const
  global_dim = b[0].size();

  assert(global_dim > 0);

  Intrepid::Matrix<RealType>
  DbDp(local_dim, global_dim);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < global_dim; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<RealType>
  DbDx(local_dim);

  // extract the jacobian
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < local_dim; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<RealType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < local_dim; ++i) {
    x(i).resize(global_dim);
    for (auto j = 0; j < global_dim; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }
  return;
}

//
// DistParamDeriv
//
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

//
// Stochastic Galerkin Residual
//
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

//
// Stochastic Galerkin Tangent
//
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

//
// Multi-Point Residual
//
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

//
// Multi-Point Jacobian
//
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

//
// Multi-Point Tangent
//
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
//
}

