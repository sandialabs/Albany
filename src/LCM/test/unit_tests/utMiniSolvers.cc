//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_UnitTestHarness.hpp>
#include <MiniLinearSolver.h>
#include <MiniNonlinearSolver.h>
#include "../../utils/MiniSolvers.h"
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

} // anonymous namespace
