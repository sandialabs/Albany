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

//
// Simple test of the linear mini solver.
//
TEUCHOS_UNIT_TEST(MiniLinearSolver, LehmerMatrix)
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

//
// Define some nonlinear systems (NLS) to test nonlinear solution methods.
//
template <typename S>
class SquareRootNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  SquareRootNLS(S const c) : c_(c) {}

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == 1);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = x(0) * x(0) - c_;

    return r;
  }

private:
  S const
  c_{0.0};
};

template <typename S>
class QuadraticNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  QuadraticNLS(S const a, S const b, S const c) :  a_(a), b_(b), c_(c) {}

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == 2);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * c_ * (x(0) - a_);
    r(1) = 2.0 * c_ * (x(1) - b_);

    return r;
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};

  S const
  c_{0.0};
};

template <typename S>
class GaussianNLS  : public Intrepid::NonlinearSystem_Base<S>
{
public:

  GaussianNLS(S const a, S const b, S const c) : a_(a), b_(b), c_(c) {}

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == 2);

    Intrepid::Vector<T, N>
    r(dimension);

    T const
    xa = (x(0) - a_) * c_;

    T const
    xb = (x(1) - b_) * c_;

    T const
    e = std::exp(- xa * xa - xb * xb);

    r(0) = 2.0 * xa * e * c_;
    r(1) = 2.0 * xb * e * c_;

    return r;
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};

  S const
  c_{0.0};
};

//
// Test the solution methods by themselves.
//

//
// Square root NLS
//
TEUCHOS_UNIT_TEST(NewtonMethod, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<RealType>;

  RealType const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::NewtonMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = std::abs(norm_square(x) - square);

  TEST_COMPARE(error, <=, method.getAbsoluteTolerance());
}

TEUCHOS_UNIT_TEST(TrustRegionMethod, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<RealType>;

  RealType const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::TrustRegionMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = std::abs(norm_square(x) - square);

  TEST_COMPARE(error, <=, method.getAbsoluteTolerance());
}

TEUCHOS_UNIT_TEST(ConjugateGradientMethod, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<RealType>;

  RealType const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::ConjugateGradientMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = std::abs(norm_square(x) - square);

  TEST_COMPARE(error, <=, method.getAbsoluteTolerance());
}

TEUCHOS_UNIT_TEST(LineSearchRegularized, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<RealType>;

  RealType const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::LineSearchRegularizedMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = std::abs(norm_square(x) - square);

  TEST_COMPARE(error, <=, method.getAbsoluteTolerance());
}

//
// Quadratic NLS
//
TEUCHOS_UNIT_TEST(NewtonMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = QuadraticNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::NewtonMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(TrustRegionMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = QuadraticNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::TrustRegionMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(ConjugateGradientMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = QuadraticNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::ConjugateGradientMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(LineSearchRegularizedMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = QuadraticNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::LineSearchRegularizedMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

//
// Gaussian NLS
//
TEUCHOS_UNIT_TEST(NewtonMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = GaussianNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::NewtonMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(TrustRegionMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = GaussianNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::TrustRegionMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(ConjugateGradientMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = GaussianNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::ConjugateGradientMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

TEUCHOS_UNIT_TEST(LineSearchRegularizedMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = GaussianNLS<RealType>;

  Intrepid::Vector<RealType, dimension> const
  minimum(4.0, 3.0);

  RealType const
  scaling = 0.125;

  NLS
  nonlinear_system(minimum(0), minimum(1), scaling);

  Intrepid::LineSearchRegularizedMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  method.solve(nonlinear_system, x);

  RealType const
  error = Intrepid::norm(x - minimum) / Intrepid::norm(minimum);

  TEST_COMPARE(error, <=, method.getRelativeTolerance());
}

//
// Test the LCM nonlinear mini solver with the corresponding solution
// methods.
//
TEUCHOS_UNIT_TEST(MiniNonLinearSolverNewtonMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::NewtonMethod<NLS, ValueT, dimension>
  newton_method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(newton_method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  ValueT const
  error = std::abs(norm_square(x) - square);

  ValueT const
  absolute_tolerance = newton_method.getAbsoluteTolerance();

  TEST_COMPARE(error, <=, absolute_tolerance);
}

TEUCHOS_UNIT_TEST(MiniNonLinearSolverTrustRegionMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::TrustRegionMethod<NLS, ValueT, dimension>
  trust_region_method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(trust_region_method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  ValueT const
  error = std::abs(norm_square(x) - square);

  ValueT const
  absolute_tolerance = trust_region_method.getAbsoluteTolerance();

  TEST_COMPARE(error, <=, absolute_tolerance);
}

TEUCHOS_UNIT_TEST(MiniNonLinearSolverConjugateGradientMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::ConjugateGradientMethod<NLS, ValueT, dimension>
  conjugate_gradient_method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(conjugate_gradient_method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  ValueT const
  error = std::abs(norm_square(x) - square);

  ValueT const
  absolute_tolerance = conjugate_gradient_method.getAbsoluteTolerance();

  TEST_COMPARE(error, <=, absolute_tolerance);
}

TEUCHOS_UNIT_TEST(MiniNonLinearSolverLineSearchRegularizedMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::LineSearchRegularizedMethod<NLS, ValueT, dimension>
  conjugate_gradient_method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(conjugate_gradient_method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  ValueT const
  error = std::abs(norm_square(x) - square);

  ValueT const
  absolute_tolerance = conjugate_gradient_method.getAbsoluteTolerance();

  TEST_COMPARE(error, <=, absolute_tolerance);
}

//
// Define some nonlinear functions (NLF) to test nonlinear optimization methods.
//
template <typename S>
class CubicFn
{
public:

  CubicFn(S const c) : c_(c) {}

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  T
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == 1);

    T
    f = x(0) * x(0) * x(0) / 3.0 - c_ * x(0);

    return f;
  }

private:
  S const
  c_{0.0};
};


} // anonymous namespace
