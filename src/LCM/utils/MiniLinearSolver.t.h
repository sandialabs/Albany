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

} // namespace LCM
