//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_UnitTestHarness.hpp>
#include <MiniLinearSolver.h>
#include <MiniNonlinearSolver.h>
#include "PHAL_AlbanyTraits.hpp"

namespace
{

TEUCHOS_UNIT_TEST(MiniLinearSolver, Instantiation)
{
  Intrepid::Index const
  dimension{3};

  // Lehmer matrix
  Intrepid::Tensor<RealType, dimension> const
  A(1.0, 0.5, 1.0/3.0, 0.5, 1.0, 2.0/3.0, 1.0/3.0, 2.0/3.0, 1.0);

  // RHS
  Intrepid::Vector<RealType, dimension> const
  b(2.0, 1.0, 1.0);

  // Known solution
  Intrepid::Vector<RealType, dimension> const
  v(2.0, -2.0/5.0, 3.0/5.0);

  Intrepid::Vector<RealType, dimension>
  x(0.0, 0.0, 0.0);

  LCM::MiniLinearSolver<PHAL::AlbanyTraits::Residual, dimension>
  solver;

  solver.solve(A, b, x);

  RealType const
  error = norm(x - v) / norm(v);

  TEST_COMPARE(error, <=, Intrepid::machine_epsilon<RealType>());
}

template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class GaussianResidual : public LCM::Residual_Base<T, N>
{
public:

  using ValueT = typename Sacado::ValueType<T>::type;

  GaussianResidual(
      ValueT const a,
      ValueT const b,
      ValueT const c) : a_(a), b_(b), c_{c} {}

  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) override
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    Intrepid::Vector<T, N>
    r(dimension);

    T const
    xa = (x(0) - a_) / c_;

    T const
    xb = (x(1) - b_) / c_;

    T const
    e = std::exp(- xa * xa - xb * xb);

    r(0) = 2.0 * xa * e / c_;
    r(1) = 2.0 * xb * e / c_;

    return r;
  }


private:
  ValueT const
  a_{0.0};

  ValueT const
  b_{0.0};

  ValueT const
  c_{0.0};

};

template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class QuadraticResidual : public LCM::Residual_Base<T, N>
{
public:

  using ValueT = typename Sacado::ValueType<T>::type;

  QuadraticResidual(ValueT const c) : c_(c) {}

  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) override
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * c_ * x(0);
    r(1) = 2.0 * c_ * x(1);

    return r;
  }


private:
  ValueT const
  c_{0.0};

};

TEUCHOS_UNIT_TEST(MiniNewtonSolver, Instantiation)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{2};

  Intrepid::Vector<ValueT, dimension>
  solution(2.0, 1.0);

  using GR = GaussianResidual<FadT, dimension>;

  GR
  residual(solution(0), solution(1), 10.0);

  LCM::NewtonSolver<PHAL::AlbanyTraits::Residual, GR, dimension>
  solver;

  Intrepid::Vector<FadT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = FadT(dimension, i, 0.0);
  }

  solver.solve(residual, x);

  Intrepid::Vector<ValueT, dimension>
  x_val = Sacado::Value<Intrepid::Vector<FadT, dimension>>::eval(x);

  ValueT const
  error = norm(x_val - solution) / norm(solution);

  TEST_COMPARE(error, <=, solver.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(MiniTrustRegionSolver, Instantiation)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{2};

  Intrepid::Vector<ValueT, dimension>
  solution(2.0, 1.0);

  using GR = GaussianResidual<FadT, dimension>;

  GR
  residual(solution(0), solution(1), 10.0);

  LCM::TrustRegionSolver<PHAL::AlbanyTraits::Residual, GR, dimension>
  solver;

  Intrepid::Vector<FadT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = FadT(dimension, i, 0.0);
  }

  solver.solve(residual, x);

  Intrepid::Vector<ValueT, dimension>
  x_val = Sacado::Value<Intrepid::Vector<FadT, dimension>>::eval(x);

  ValueT const
  error = norm(x_val - solution) / norm(solution);

  TEST_COMPARE(error, <=, solver.getRelativeTolerance());
}

} // anonymous namespace
