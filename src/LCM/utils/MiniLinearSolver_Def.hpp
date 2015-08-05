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
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>::MiniLinearSolver()
: MiniLinearSolver_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
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
// Stochastic Galerkin Residual
//
#ifdef ALBANY_SG
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::SGResidual, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Stochastic Galerkin types yet\n");
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Stochastic Galerkin types yet\n");
  return;
}

//
// Stochastic Galerkin Jacobian
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::SGJacobian, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Stochastic Galerkin types yet\n");
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Stochastic Galerkin types yet\n");
  return;
}

//
// Stochastic Galerkin Tangent
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::SGTangent, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Stochastic Galerkin types yet\n");
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Stochastic Galerkin types yet\n");
  return;
}
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE 
//
// Multi-Point Residual
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::MPResidual, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Multi-Point types yet\n");
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Multi-Point types yet\n");
  return;
}

//
// Multi-Point Jacobian
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::MPJacobian, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Multi-Point types yet\n");
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Multi-Point types yet\n");
  return;
}

//
// Multi-Point Tangent
//
template<typename Traits>
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits>::MiniLinearSolver():
MiniLinearSolver_Base<PHAL::AlbanyTraits::MPTangent, Traits>()
{
  return;
}

template<typename Traits>
template <Intrepid::Index N = Intrepid::DYNAMIC>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Multi-Point types yet\n");
  return;
}

template<typename Traits>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits>::
computeFadInfo(
    Intrepid::Tensor<ScalarT> const & A,
    Intrepid::Vector<ScalarT> const & b,
    Intrepid::Vector<ScalarT> & x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "MiniLinearSolver does not have Multi-Point types yet\n");
  return;
}
#endif // ALBANY_ENSEMBLE

} // namespace LCM
