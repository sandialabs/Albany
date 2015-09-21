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

// Test one system with one method.
template <typename NLS, typename NLM, typename T, Intrepid::Index N>
bool
solveNLSwithNLM(NLS & system, NLM & method, Intrepid::Vector<T, N> & x)
{
  method.solve(system, x);
  method.printReport(std::cout);

  return method.isConverged();
}

// Test one system with various methods.
template <typename NLS, typename T, Intrepid::Index N>
bool
solveNLS(NLS & system, Intrepid::Vector<T, N> const & x)
{
  bool
  all_ok = true;

  Intrepid::Vector<T, N>
  y;

  Intrepid::NewtonMethod<NLS, T, N>
  newton;

  y = x;

  bool const
  newton_ok = solveNLSwithNLM(system, newton, y);

  all_ok = all_ok && newton_ok;

  Intrepid::TrustRegionMethod<NLS, T, N>
  trust_region;

  y = x;

  bool const
  trust_region_ok = solveNLSwithNLM(system, trust_region, y);

  all_ok = all_ok && trust_region_ok;

  Intrepid::ConjugateGradientMethod<NLS, T, N>
  pcg;

  y = x;

  bool const
  pcg_ok = solveNLSwithNLM(system, pcg, y);

  all_ok = all_ok && pcg_ok;

  Intrepid::LineSearchRegularizedMethod<NLS, T, N>
  line_search;

  y = x;

  bool const
  line_search_ok = solveNLSwithNLM(system, line_search, y);

  all_ok = all_ok && line_search_ok;

  return all_ok;
}

// Test various systems with various methods.
bool testSystemsAndMethods()
{
  bool
  all_ok = true;

  Intrepid::Vector<RealType>
  x;

  LCM::SquareRootNLS<RealType>
  square_root(2.0);

  x.set_dimension(LCM::SquareRootNLS<RealType>::DIMENSION);

  x(0) = 10.0;

  bool const
  square_root_ok = solveNLS(square_root, x);

  all_ok = all_ok && square_root_ok;

  LCM::QuadraticNLS<RealType>
  quadratic(10.0, 15.0, 1.0);

  x.set_dimension(LCM::QuadraticNLS<RealType>::DIMENSION);

  x(0) = -15.0;
  x(1) = -10.0;

  bool const
  quadratic_ok = solveNLS(quadratic, x);

  all_ok = all_ok && quadratic_ok;

  LCM::GaussianNLS<RealType>
  gaussian(1.0, 2.0, 0.125);

  x.set_dimension(LCM::GaussianNLS<RealType>::DIMENSION);

  x(0) = 0.0;
  x(1) = 0.0;

  bool const
  gaussian_ok = solveNLS(gaussian, x);

  all_ok = all_ok && gaussian_ok;

  LCM::BananaNLS<RealType>
  banana;

  x.set_dimension(LCM::BananaNLS<RealType>::DIMENSION);

  x(0) = 0.0;
  x(1) = 3.0;

  bool const
  banana_ok = solveNLS(banana, x);

  all_ok = all_ok && banana_ok;

  LCM::MatyasNLS<RealType>
  matyas;

  x.set_dimension(LCM::MatyasNLS<RealType>::DIMENSION);

  x(0) = 10.0;
  x(1) =  0.0;

  bool const
  matyas_ok = solveNLS(matyas, x);

  all_ok = all_ok && matyas_ok;

  LCM::McCormickNLS<RealType>
  mccormick;

  x.set_dimension(LCM::McCormickNLS<RealType>::DIMENSION);

  x(0) = -0.5;
  x(1) = -1.5;

  bool const
  mccormick_ok = solveNLS(mccormick, x);

  all_ok = all_ok && mccormick_ok;

  LCM::StyblinskiTangNLS<RealType>
  styblinski_tang;

  x.set_dimension(LCM::StyblinskiTangNLS<RealType>::DIMENSION);

  x(0) = -4.0;
  x(1) = -4.0;

  bool const
  styblinski_tang_ok = solveNLS(styblinski_tang, x);

  all_ok = all_ok && styblinski_tang_ok;

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
// Test the LCM nonlinear mini solver.
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

} // anonymous namespace
