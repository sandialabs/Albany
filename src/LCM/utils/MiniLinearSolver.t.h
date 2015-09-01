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
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Residual, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  x = Intrepid::solve(A, b);
  return;
}

//
// Jacobian
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<ValueT, N>
  Df(dimension);

  Intrepid::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<ValueT, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Tensor<ValueT>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  computeFADInfo(b, DbDx, x);

  return;
}

//
// Tangent
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<ValueT, N>
  Df(dimension);

  Intrepid::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<ValueT, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Matrix<ValueT>
  DbDp(dimension, order);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<ValueT>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<ValueT>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

//
// DistParamDeriv
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<ValueT, N>
  Df(dimension);

  Intrepid::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<ValueT, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Matrix<ValueT>
  DbDp(dimension, order);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<ValueT>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<ValueT>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

#ifdef ALBANY_SG
//
// SGResidual
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  x = Intrepid::solve(A, b);
  return;
}

//
// SGJacobian
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<SGType, N>
  Df(dimension);

  Intrepid::Tensor<SGType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<SGType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Matrix<SGType>
  DbDp(dimension, order);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<SGType>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<SGType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

//
// SGTangent
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<SGType, N>
  Df(dimension);

  Intrepid::Tensor<SGType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<SGType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Matrix<SGType>
  DbDp(dimension, order);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<SGType>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<SGType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  x = Intrepid::solve(A, b);
  return;
}

//
// MPJacobian
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<MPType, N>
  Df(dimension);

  Intrepid::Tensor<MPType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<MPType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Matrix<MPType>
  DbDp(dimension, order);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<MPType>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<MPType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

//
// MPTangent
//
template<Intrepid::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, N>::
solve(
    Intrepid::Tensor<ScalarT, N> const & A,
    Intrepid::Vector<ScalarT, N> const & b,
    Intrepid::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid::Vector<MPType, N>
  Df(dimension);

  Intrepid::Tensor<MPType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid::Vector<MPType, N> const
  Dx = Intrepid::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  auto const
  order = b[0].size();

  assert(order > 0);

  Intrepid::Matrix<MPType>
  DbDp(dimension, order);

  // extract sensitivities of objective function(s) wrt p
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DbDp(i, j) = b(i).dx(j);
    }
  }

  Intrepid::Tensor<MPType>
  DbDx(dimension);

  // extract the jacobian
  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < dimension; ++j) {
      DbDx(i, j) = A(i, j).val();
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<MPType>
  DxDp = Intrepid::solve(DbDx, DbDp);

  // Unpack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  return;
}

#endif // ALBANY_ENSEMBLE

} // namespace LCM
