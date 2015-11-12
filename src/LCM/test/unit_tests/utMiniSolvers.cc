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
// Test the solution methods by themselves.
//

// Test one function with one method.
template <typename STEP, typename FN, typename T, Intrepid::Index N>
bool
solveFNwithSTEP(STEP & step_method, FN & function, Intrepid::Vector<T, N> & x)
{
  Intrepid::Minimizer<T, N>
  minimizer;

  minimizer.solve(step_method, function, x);

  minimizer.printReport(std::cout);

  return minimizer.converged;
}

// Test one system with various methods.
template <typename FN, typename T, Intrepid::Index N>
bool
solveFN(FN & function, Intrepid::Vector<T, N> const & x)
{
  bool
  all_ok = true;

  Intrepid::Vector<T, N>
  y;

  Intrepid::NewtonStep<T, N>
  newton_step;

  y = x;

  bool const
  newton_ok = solveFNwithSTEP(newton_step, function, y);

  all_ok = all_ok && newton_ok;

  Intrepid::TrustRegionStep<T, N>
  trust_region_step;

  y = x;

  bool const
  trust_region_ok = solveFNwithSTEP(trust_region_step, function, y);

  all_ok = all_ok && trust_region_ok;

  Intrepid::ConjugateGradientStep<T, N>
  pcg_step;

  y = x;

  bool const
  pcg_ok = solveFNwithSTEP(pcg_step, function, y);

  all_ok = all_ok && pcg_ok;

  Intrepid::LineSearchRegularizedStep<T, N>
  line_search_step;

  y = x;

  bool const
  line_search_ok = solveFNwithSTEP(line_search_step, function, y);

  all_ok = all_ok && line_search_ok;

  return all_ok;
}

// Test various systems with various methods.
bool testSystemsAndMethods()
{
  constexpr Intrepid::Index
  max_dimension{2};

  bool
  all_ok = true;

  Intrepid::Vector<RealType, max_dimension>
  x;

  LCM::SquareRootNLS<RealType>
  square_root(2.0);

  x.set_dimension(LCM::SquareRootNLS<RealType>::DIMENSION);

  x(0) = 10.0;

  bool const
  square_root_ok = solveFN(square_root, x);

  all_ok = all_ok && square_root_ok;

  LCM::QuadraticNLS<RealType>
  quadratic(10.0, 15.0, 1.0);

  x.set_dimension(LCM::QuadraticNLS<RealType>::DIMENSION);

  x(0) = -15.0;
  x(1) = -10.0;

  bool const
  quadratic_ok = solveFN(quadratic, x);

  all_ok = all_ok && quadratic_ok;

  LCM::GaussianNLS<RealType>
  gaussian(1.0, 2.0, 0.125);

  x.set_dimension(LCM::GaussianNLS<RealType>::DIMENSION);

  x(0) = 0.0;
  x(1) = 0.0;

  bool const
  gaussian_ok = solveFN(gaussian, x);

  all_ok = all_ok && gaussian_ok;

  LCM::BananaNLS<RealType>
  banana;

  x.set_dimension(LCM::BananaNLS<RealType>::DIMENSION);

  x(0) = 0.0;
  x(1) = 3.0;

  bool const
  banana_ok = solveFN(banana, x);

  all_ok = all_ok && banana_ok;

  LCM::MatyasNLS<RealType>
  matyas;

  x.set_dimension(LCM::MatyasNLS<RealType>::DIMENSION);

  x(0) = 10.0;
  x(1) =  0.0;

  bool const
  matyas_ok = solveFN(matyas, x);

  all_ok = all_ok && matyas_ok;

  LCM::McCormickNLS<RealType>
  mccormick;

  x.set_dimension(LCM::McCormickNLS<RealType>::DIMENSION);

  x(0) = -0.5;
  x(1) = -1.5;

  bool const
  mccormick_ok = solveFN(mccormick, x);

  all_ok = all_ok && mccormick_ok;

  LCM::StyblinskiTangNLS<RealType>
  styblinski_tang;

  x.set_dimension(LCM::StyblinskiTangNLS<RealType>::DIMENSION);

  x(0) = -4.0;
  x(1) = -4.0;

  bool const
  styblinski_tang_ok = solveFN(styblinski_tang, x);

  all_ok = all_ok && styblinski_tang_ok;

  LCM::Paraboloid<RealType>
  paraboloid;

  x.set_dimension(LCM::Paraboloid<RealType>::DIMENSION);

  x(0) = 128.0;
  x(1) = 256.0;;

  bool const
  paraboloid_ok = solveFN(paraboloid, x);

  all_ok = all_ok && paraboloid_ok;

  LCM::Beale<RealType>
  beale;

  x.set_dimension(LCM::Beale<RealType>::DIMENSION);

  x(0) = -4.5;
  x(1) = -4.5;

  bool const
  beale_ok = solveFN(beale, x);

  all_ok = all_ok && beale_ok;

  LCM::Booth<RealType>
  booth;

  x.set_dimension(LCM::Booth<RealType>::DIMENSION);

  x(0) = -10.0;
  x(1) = -10.0;

  bool const
  booth_ok = solveFN(booth, x);

  all_ok = all_ok && booth_ok;

  LCM::GoldsteinPrice<RealType>
  goldstein_price;

  x.set_dimension(LCM::GoldsteinPrice<RealType>::DIMENSION);

  x(0) = 2.0;
  x(1) = 2.0;

  bool const
  goldstein_price_ok = solveFN(goldstein_price, x);

  all_ok = all_ok && goldstein_price_ok;

  return all_ok;
}

TEUCHOS_UNIT_TEST(NonlinearSystems, NonlinearMethods)
{
  bool const
  passed = testSystemsAndMethods();

  TEST_COMPARE(passed, ==, true);
}

//
// Simple test of the linear mini solver.
//
TEUCHOS_UNIT_TEST(MiniLinearSolver, LehmerMatrix)
{
  constexpr Intrepid::Index
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

TEUCHOS_UNIT_TEST(Testing, OptimizationMethods)
{
  constexpr Intrepid::Index
  dimension{2};

  LCM::BananaNLS<RealType>
  banana;

  Intrepid::NewtonStep<RealType, dimension>
  step;

  Intrepid::Minimizer<RealType, dimension>
  minimizer;

  Intrepid::Vector<RealType, dimension>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  minimizer.solve(step, banana, x);

  minimizer.printReport(std::cout);

  TEST_COMPARE(true, ==, true);
}

//
// Test the LCM mini minimizer.
//
TEUCHOS_UNIT_TEST(AlbanyResidual, NewtonBanana)
{
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  constexpr
  Intrepid::Index
  dimension{2};

  LCM::BananaNLS<ScalarT>
  banana;

  Intrepid::NewtonStep<ScalarT, dimension>
  step;

  Intrepid::Minimizer<ScalarT, dimension>
  minimizer;

  Intrepid::Vector<ScalarT, dimension>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::miniMinimize(minimizer, step, banana, x);

  minimizer.printReport(std::cout);

  TEST_COMPARE(minimizer.converged, ==, true);
}

//
// Test the LCM mini minimizer.
//
TEUCHOS_UNIT_TEST(AlbanyJacobian, NewtonBanana)
{
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  constexpr
  Intrepid::Index
  dimension{2};

  LCM::BananaNLS<ScalarT>
  banana;

  Intrepid::NewtonStep<ScalarT, dimension>
  step;

  Intrepid::Minimizer<ScalarT, dimension>
  minimizer;

  Intrepid::Vector<ScalarT, dimension>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::miniMinimize(minimizer, step, banana, x);

  minimizer.printReport(std::cout);

  TEST_COMPARE(minimizer.converged, ==, true);
}

TEUCHOS_UNIT_TEST(Testing, ValueGradientHessian)
{
  constexpr Intrepid::Index
  dimension{2};

  LCM::Paraboloid<RealType>
  p;

  Intrepid::Vector<RealType, dimension>
  x(0.0, 0.0);

  std::cout << "Point   : " << x << '\n';
  std::cout << "Value   : " << p.value(x) << '\n';
  std::cout << "Gradient: " << p.gradient(x) << '\n';
  std::cout << "Hessian : " << p.hessian(x) << '\n';

  TEST_COMPARE(true, ==, true);
}

TEUCHOS_UNIT_TEST(Testing, MixedStorage)
{
  Intrepid::Index const
  dimension{2};

  std::cout << '\n';

  Intrepid::Vector<RealType, 3>
  v(1.0, 2.0, 3.0);

  v.set_dimension(dimension);

  std::cout << "Vector   : " << v << '\n';

  Intrepid::Tensor<RealType, 3>
  A(1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0);

  A.set_dimension(dimension);

  std::cout << "Tensor   : " << A << '\n';

  Intrepid::Matrix<RealType, 3, 4>
  B(Intrepid::ONES);

  B.set_dimensions(4, 2);

  std::cout << "Matrix   : " << B << '\n';

  TEST_COMPARE(true, ==, true);
}

} // anonymous namespace
