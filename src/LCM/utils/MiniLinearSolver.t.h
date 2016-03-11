//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Residual, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  x = Intrepid2::solve(A, b);
  return;
}

//
// Jacobian
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<ValueT, N>
  Df(dimension);

  Intrepid2::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<ValueT, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<ValueT>
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
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<ValueT, N>
  Df(dimension);

  Intrepid2::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<ValueT, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<ValueT>
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
// DistParamDeriv
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<ValueT, N>
  Df(dimension);

  Intrepid2::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<ValueT, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<ValueT>
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

#ifdef ALBANY_SG
//
// SGResidual
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  x = Intrepid2::solve(A, b);
  return;
}

//
// SGJacobian
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<SGType, N>
  Df(dimension);

  Intrepid2::Tensor<SGType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<SGType, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<SGType>
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
// SGTangent
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<SGType, N>
  Df(dimension);

  Intrepid2::Tensor<SGType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<SGType, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<SGType>
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

#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  x = Intrepid2::solve(A, b);
  return;
}

//
// MPJacobian
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<MPType, N>
  Df(dimension);

  Intrepid2::Tensor<MPType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<MPType, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<MPType>
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
// MPTangent
//
template<Intrepid2::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, N>::
solve(
    Intrepid2::Tensor<ScalarT, N> const & A,
    Intrepid2::Vector<ScalarT, N> const & b,
    Intrepid2::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  Intrepid2::Vector<MPType, N>
  Df(dimension);

  Intrepid2::Tensor<MPType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  Intrepid2::Vector<MPType, N> const
  Dx = Intrepid2::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  Intrepid2::Tensor<MPType>
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

#endif // ALBANY_ENSEMBLE

} // namespace LCM
