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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Residual, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  x = minitensor::solve(A, b);
  return;
}

//
// Jacobian
//
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Jacobian, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<ValueT, N>
  Df(dimension);

  minitensor::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<ValueT, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<ValueT>
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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::Tangent, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<ValueT, N>
  Df(dimension);

  minitensor::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<ValueT, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<ValueT>
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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::DistParamDeriv, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<ValueT, N>
  Df(dimension);

  minitensor::Tensor<ValueT, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<ValueT, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<ValueT>
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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGResidual, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  x = minitensor::solve(A, b);
  return;
}

//
// SGJacobian
//
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGJacobian, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<SGType, N>
  Df(dimension);

  minitensor::Tensor<SGType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<SGType, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<SGType>
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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::SGTangent, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<SGType, N>
  Df(dimension);

  minitensor::Tensor<SGType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<SGType, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<SGType>
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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPResidual, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  x = minitensor::solve(A, b);
  return;
}

//
// MPJacobian
//
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPJacobian, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<MPType, N>
  Df(dimension);

  minitensor::Tensor<MPType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<MPType, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<MPType>
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
template<minitensor::Index N>
void
MiniLinearSolver<PHAL::AlbanyTraits::MPTangent, N>::
solve(
    minitensor::Tensor<ScalarT, N> const & A,
    minitensor::Vector<ScalarT, N> const & b,
    minitensor::Vector<ScalarT, N> & x)
{
  //
  // First deal with values
  //
  auto const
  dimension = b.get_dimension();

  minitensor::Vector<MPType, N>
  Df(dimension);

  minitensor::Tensor<MPType, N>
  DfDx(dimension);

  for (auto i = 0; i < dimension; ++i) {
    Df(i) = b(i).val();

    for (auto j = 0; j < dimension; ++j) {
      DfDx(i, j) = A(i, j).val();
    }
  }

  minitensor::Vector<MPType, N> const
  Dx = minitensor::solve(DfDx, Df);

  for (auto i = 0; i < dimension; ++i) {
    x(i).val() = Dx(i);
  }

  //
  // Then deal with derivatives
  //
  minitensor::Tensor<MPType>
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
