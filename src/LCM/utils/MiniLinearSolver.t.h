//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// Specializations
//

//
// Residual
//
template<typename Traits, Intrepid::Index N>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  x = Intrepid::solve(A, b);
  return;
}

template<typename Traits, Intrepid::Index N>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::Residual, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  return;
}

//
// Jacobian
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<RealType, N>
  Df(local_dim);

  Intrepid::Tensor<RealType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<RealType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}

//
// Tangent
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<RealType, N>
  Df(local_dim);

  Intrepid::Tensor<RealType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<RealType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}

//
// DistParamDeriv
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<RealType, N>
  Df(local_dim);

  Intrepid::Tensor<RealType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<RealType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
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
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}

#ifdef ALBANY_SG
//
// SGResidual
//
template<typename Traits, Intrepid::Index N>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  x = Intrepid::solve(A, b);
  return;
}

template<typename Traits, Intrepid::Index N>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  return;
}

//
// SGJacobian
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<SGType, N>
  Df(local_dim);

  Intrepid::Tensor<SGType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<SGType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  auto const
  global_dim = b[0].size();

  assert(global_dim > 0);

  Intrepid::Matrix<SGType>
  DbDp(local_dim, global_dim);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < global_dim; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<SGType>
  DbDx(local_dim);

  // extract the jacobian
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < local_dim; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<SGType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < local_dim; ++i) {
    x(i).resize(global_dim);
    for (auto j = 0; j < global_dim; ++j) {
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}

//
// SGTangent
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<SGType, N>
  Df(local_dim);

  Intrepid::Tensor<SGType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<SGType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  auto const
  global_dim = b[0].size();

  assert(global_dim > 0);

  Intrepid::Matrix<SGType>
  DbDp(local_dim, global_dim);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < global_dim; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<SGType>
  DbDx(local_dim);

  // extract the jacobian
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < local_dim; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<SGType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < local_dim; ++i) {
    x(i).resize(global_dim);
    for (auto j = 0; j < global_dim; ++j) {
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}
#endif

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<typename Traits, Intrepid::Index N>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  x = Intrepid::solve(A, b);
  return;
}

template<typename Traits, Intrepid::Index N>
void
inline
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  return;
}

//
// MPJacobian
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<MPType, N>
  Df(local_dim);

  Intrepid::Tensor<MPType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<MPType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  auto const
  global_dim = b[0].size();

  assert(global_dim > 0);

  Intrepid::Matrix<MPType>
  DbDp(local_dim, global_dim);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < global_dim; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<MPType>
  DbDx(local_dim);

  // extract the jacobian
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < local_dim; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<MPType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < local_dim; ++i) {
    x(i).resize(global_dim);
    for (auto j = 0; j < global_dim; ++j) {
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}

//
// MPTangent
//
template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  Intrepid::Vector<MPType, N>
  Df(local_dim);

  Intrepid::Tensor<MPType, N>
  DfDx(local_dim);

  for (auto i = 0; i < local_dim; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < local_dim; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<MPType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < local_dim; ++i) {
    x(i).val() = Dx(i);
  }

  return;
}

template<typename Traits, Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, Traits, N>::
computeFadInfo(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  auto const
  local_dim = b.get_dimension();

  auto const
  global_dim = b[0].size();

  assert(global_dim > 0);

  Intrepid::Matrix<MPType>
  DbDp(local_dim, global_dim);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < global_dim; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<MPType>
  DbDx(local_dim);

  // extract the jacobian
  for (auto i = 0; i < local_dim; ++i) {
    for (auto j = 0; j < local_dim; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<MPType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < local_dim; ++i) {
    x(i).resize(global_dim);
    for (auto j = 0; j < global_dim; ++j) {
      x(i).fastAccessDx(j) = DxDp(i, j);
    }
  }

  return;
}
#endif

} // namespace LCM
