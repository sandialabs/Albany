//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_UnitTestHarness.hpp>
#include <MiniLinearSolver.h>
#include <MiniNonlinearSolver.h>
#include "PHAL_AlbanyTraits.hpp"

//
// Define some nonlinear systems (NLS) to test nonlinear solution methods.
//
namespace LCM {

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

template <typename S>
class BananaNLS  : public Intrepid::NonlinearSystem_Base<S>
{
public:

  BananaNLS(S const a, S const b) : a_(a), b_(b) {}

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == 2);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2 * (x(0) - a_) + 4 * b_ * x(0) * (x(0) * x(0) - x(1));
    r(1) = 2 * b_ * (x(1) - x(0) * x(0));

    return r;
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};
};

} // namespace LCM

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
// Test the solution methods by themselves.
//

//
// Square root NLS
//
TEUCHOS_UNIT_TEST(IntrepidNewtonMethod, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidTrustRegionMethod, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidConjugateGradientMethod, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidLineSearchRegularized, SquareRoot)
{
  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

//
// Quadratic NLS
//
TEUCHOS_UNIT_TEST(IntrepidNewtonMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::QuadraticNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidTrustRegionMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::QuadraticNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidConjugateGradientMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::QuadraticNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidLineSearchRegularizedMethod, Quadratic)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::QuadraticNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

//
// Gaussian NLS
//
TEUCHOS_UNIT_TEST(IntrepidNewtonMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::GaussianNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidTrustRegionMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::GaussianNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidConjugateGradientMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::GaussianNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidLineSearchRegularizedMethod, Gaussian)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::GaussianNLS<RealType>;

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
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

//
// Banana function
//
TEUCHOS_UNIT_TEST(IntrepidNewtonMethod, Banana)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::BananaNLS<RealType>;

  RealType const
  a = 1.0;

  RealType const
  b = 100.0;

  Intrepid::Vector<RealType, dimension> const
  minimum(a, a * a);

  NLS
  nonlinear_system(a, b);

  Intrepid::NewtonMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  x(0) = 0.0;
  x(1) = 3.0;

  method.solve(nonlinear_system, x);
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidTrustRegionMethod, Banana)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::BananaNLS<RealType>;

  RealType const
  a = 1.0;

  RealType const
  b = 100.0;

  Intrepid::Vector<RealType, dimension> const
  minimum(a, a * a);

  NLS
  nonlinear_system(a, b);

  Intrepid::TrustRegionMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  x(0) = 0.0;
  x(1) = 3.0;

  method.solve(nonlinear_system, x);
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidConjugateGradientMethod, Banana)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::BananaNLS<RealType>;

  RealType const
  a = 1.0;

  RealType const
  b = 100.0;

  Intrepid::Vector<RealType, dimension> const
  minimum(a, a * a);

  NLS
  nonlinear_system(a, b);

  Intrepid::ConjugateGradientMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  x(0) = 0.0;
  x(1) = 3.0;

  method.solve(nonlinear_system, x);
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(IntrepidLineSearchRegularizedMethod, Banana)
{
  Intrepid::Index const
  dimension{2};

  using NLS = LCM::BananaNLS<RealType>;

  RealType const
  a = 1.0;

  RealType const
  b = 100.0;

  Intrepid::Vector<RealType, dimension> const
  minimum(a, a * a);

  NLS
  nonlinear_system(a, b);

  Intrepid::LineSearchRegularizedMethod<NLS, RealType, dimension>
  method;

  Intrepid::Vector<RealType, dimension>
  x;

  // Initial guess
  x(0) = 0.0;
  x(1) = 3.0;

  method.solve(nonlinear_system, x);
  method.printReport(std::cout);

  TEST_COMPARE(method.isConverged(), ==, true);
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

  using NLS = LCM::SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::NewtonMethod<NLS, ValueT, dimension>
  method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(MiniNonLinearSolverTrustRegionMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::TrustRegionMethod<NLS, ValueT, dimension>
  method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(MiniNonLinearSolverConjugateGradientMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::ConjugateGradientMethod<NLS, ValueT, dimension>
  method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  TEST_COMPARE(method.isConverged(), ==, true);
}

TEUCHOS_UNIT_TEST(MiniNonLinearSolverLineSearchRegularizedMethod, SquareRoot)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;
  using FadT = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Index const
  dimension{1};

  using NLS = LCM::SquareRootNLS<ValueT>;

  ValueT const
  square = 2.0;

  NLS
  nonlinear_system(square);

  Intrepid::LineSearchRegularizedMethod<NLS, ValueT, dimension>
  method;

  LCM::MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, dimension>
  solver(method);

  Intrepid::Vector<ScalarT, dimension>
  x;

  // Initial guess
  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x(i) = 1.0;
  }

  solver.solve(nonlinear_system, x);

  TEST_COMPARE(method.isConverged(), ==, true);
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
