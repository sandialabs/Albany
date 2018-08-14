//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MiniNonlinearSolver.h"
#include "MiniSolvers.h"
#include "MiniTensor_FunctionSet.h"
#include "ROL_MiniTensor_MiniSolver.hpp"
#include "gtest/gtest.h"

int
main(int ac, char* av[])
{
  Kokkos::initialize();

  ::testing::GTEST_FLAG(print_time) = (ac > 1) ? true : false;

  ::testing::InitGoogleTest(&ac, av);

  auto const retval = RUN_ALL_TESTS();

  Kokkos::finalize();

  return retval;
}

//
// Test the LCM ROL mini minimizer.
//
namespace {

template <typename EvalT>
bool
bananaRosenbrock()
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using FN  = LCM::Banana_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;

  ValueT const a = 1.0;

  ValueT const b = 100.0;

  // Function to optimize
  FN fn(a, b);

  MIN minimizer;

  minimizer.verbose = print_output;

  // Define algorithm.
  std::string const algoname{"Line Search"};

  // Set parameters.
  Teuchos::ParameterList params;

  params.sublist("Step")
      .sublist("Line Search")
      .sublist("Descent Method")
      .set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolverROL<MIN, FN, EvalT, DIM> mini_solver(
      minimizer, algoname, params, fn, x);

  minimizer.printReport(os);

  return minimizer.converged;
}

}  // namespace

TEST(Rosenbrock, AlbanyResidualROL)
{
  bool const converged = bananaRosenbrock<PHAL::AlbanyTraits::Residual>();

  ASSERT_EQ(converged, true);
}

TEST(Rosenbrock, AlbanyJacobianROL)
{
  bool const converged = bananaRosenbrock<PHAL::AlbanyTraits::Jacobian>();

  ASSERT_EQ(converged, true);
}

namespace {

template <typename EvalT>
bool
paraboloid()
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using FN  = LCM::Paraboloid_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;

  ValueT const a = 0.0;

  ValueT const b = 0.0;

  // Function to optimize
  FN fn(a, b);

  MIN minimizer;

  minimizer.verbose = print_output;

  // Define algorithm.
  std::string const algoname{"Line Search"};

  // Set parameters.
  Teuchos::ParameterList params;

  params.sublist("Step")
      .sublist("Line Search")
      .sublist("Descent Method")
      .set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 10.0 * minitensor::random<ValueT>();
  x(1) = 10.0 * minitensor::random<ValueT>();

  LCM::MiniSolverROL<MIN, FN, EvalT, DIM> mini_solver(
      minimizer, algoname, params, fn, x);

  minimizer.printReport(os);

  return minimizer.converged;
}

}  // anonymous namespace

TEST(Paraboloid, PlainROLResidual)
{
  bool const converged = paraboloid<PHAL::AlbanyTraits::Residual>();

  ASSERT_EQ(converged, true);
}

TEST(Paraboloid, PlainROLJacobian)
{
  bool const converged = paraboloid<PHAL::AlbanyTraits::Jacobian>();

  ASSERT_EQ(converged, true);
}

namespace {

template <typename EvalT>
bool
paraboloidBounds()
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using FN  = LCM::Paraboloid_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;
  using BC  = minitensor::Bounds<ValueT, DIM>;

  ValueT const a = 0.0;

  ValueT const b = 0.0;

  // Function to optimize
  FN fn(a, b);

  MIN minimizer;

  minimizer.verbose = print_output;

  // (1, -\infty)
  minitensor::Vector<ValueT, DIM> lo(1.0, ROL::ROL_NINF<ValueT>());

  // 10, +\infty)
  minitensor::Vector<ValueT, DIM> hi(10.0, ROL::ROL_INF<ValueT>());

  // Constraint that defines the feasible region
  BC bounds(lo, hi);

  // Define algorithm.
  std::string const algoname{"Line Search"};

  // Set parameters.
  Teuchos::ParameterList params;

  params.sublist("Step")
      .sublist("Line Search")
      .sublist("Descent Method")
      .set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 10.0 * minitensor::random<ValueT>();
  x(1) = 10.0 * minitensor::random<ValueT>();

  LCM::MiniSolverBoundsROL<MIN, FN, BC, EvalT, DIM> mini_solver(
      minimizer, algoname, params, fn, bounds, x);

  minimizer.printReport(os);

  return minimizer.converged;
}

}  // anonymous namespace

TEST(Paraboloid, BoundsROLResidual)
{
  bool const converged = paraboloidBounds<PHAL::AlbanyTraits::Residual>();

  ASSERT_EQ(converged, true);
}

TEST(Paraboloid, BoundsROLJacobian)
{
  bool const converged = paraboloidBounds<PHAL::AlbanyTraits::Jacobian>();

  ASSERT_EQ(converged, true);
}

namespace {

template <typename EvalT>
bool
paraboloidEquality()
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index NUM_VAR{2};

  constexpr minitensor::Index NUM_CONSTR{1};

  using FN  = LCM::Paraboloid_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, NUM_VAR>;
  using EIC = minitensor::Circumference<ValueT, NUM_CONSTR, NUM_VAR>;

  ValueT const a = 2.0;

  ValueT const b = 0.0;

  ValueT const r = 1.0;

  // Function to optimize
  FN fn;

  MIN minimizer;

  minimizer.verbose = print_output;

  // Constraint that defines the feasible region
  EIC eq_constr(r, a, b);

  // Define algorithm.
  std::string const algoname{"Composite Step"};

  // Set parameters.
  Teuchos::ParameterList params;

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Optimality System Solver")
      .set("Nominal Relative Tolerance", 1.e-8);

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Optimality System Solver")
      .set("Fix Tolerance", true);

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Tangential Subproblem Solver")
      .set("Iteration Limit", 128);

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Tangential Subproblem Solver")
      .set("Relative Tolerance", 1e-6);

  params.sublist("Step").sublist(algoname).set("Output Level", 0);
  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Constraint Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-18);
  params.sublist("Status Test").set("Iteration Limit", 128);

  // Set initial guess
  minitensor::Vector<ScalarT, NUM_VAR> x(minitensor::Filler::ONES);

  // Set constraint vector
  minitensor::Vector<ScalarT, NUM_CONSTR> c(minitensor::Filler::ZEROS);

  LCM::MiniSolverEqIneqROL<MIN, FN, EIC, EvalT, NUM_VAR, NUM_CONSTR>
      mini_solver(minimizer, algoname, params, fn, eq_constr, x, c);

  minimizer.printReport(os);

  return minimizer.converged;
}

TEST(Paraboloid, EqualityROLResidual)
{
  bool const converged = paraboloidEquality<PHAL::AlbanyTraits::Residual>();

  ASSERT_EQ(converged, true);
}

TEST(Paraboloid, EqualityROLJacobian)
{
  bool const converged = paraboloidEquality<PHAL::AlbanyTraits::Jacobian>();

  ASSERT_EQ(converged, true);
}

TEST(MiniTensor_ROL, Paraboloid_EqualityConstraint)
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  constexpr minitensor::Index NUM_VAR{2};

  constexpr minitensor::Index NUM_CONSTR{1};

  double const a = 2.0;

  double const b = 0.0;

  double const r = 1.0;

  // Function to optimize
  minitensor::Paraboloid<double, NUM_VAR> fn;

  // Constraint that defines the feasible region
  minitensor::Circumference<double, NUM_CONSTR, NUM_VAR> eq_constr(r, a, b);

  // Define algorithm.
  std::string const algoname{"Composite Step"};

  // Set parameters.
  Teuchos::ParameterList params;

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Optimality System Solver")
      .set("Nominal Relative Tolerance", 1.e-8);

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Optimality System Solver")
      .set("Fix Tolerance", true);

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Tangential Subproblem Solver")
      .set("Iteration Limit", 128);

  params.sublist("Step")
      .sublist(algoname)
      .sublist("Tangential Subproblem Solver")
      .set("Relative Tolerance", 1e-6);

  params.sublist("Step").sublist(algoname).set("Output Level", 0);
  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Constraint Tolerance", 1.0e-12);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-18);
  params.sublist("Status Test").set("Iteration Limit", 128);

  // Set initial guess
  minitensor::Vector<double, NUM_VAR> x(minitensor::Filler::ONES);

  // Set constraint vector
  minitensor::Vector<double, NUM_CONSTR> c(minitensor::Filler::ZEROS);

  ROL::MiniTensor_Minimizer<double, NUM_VAR> minimizer;

  minimizer.verbose = print_output;

  minimizer.solve(algoname, params, fn, eq_constr, x, c);

  minimizer.printReport(os);

  double const tol{1.0e-14};

  minitensor::Vector<double, NUM_VAR> soln(1.0, 0.0);

  double const error = minitensor::norm(soln - x);

  ASSERT_LE(error, tol);
}

}  // anonymous namespace
