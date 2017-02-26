//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "gtest/gtest.h"
#include "MiniSolvers.h"
#include "MiniNonlinearSolver.h"
#include "ROL_MiniTensor_MiniSolver.hpp"

int
main(int ac, char * av[])
{
  Kokkos::initialize();

  ::testing::GTEST_FLAG(print_time) = (ac > 1) ? true : false;

  ::testing::InitGoogleTest(&ac, av);

  auto const
  retval = RUN_ALL_TESTS();

  Kokkos::finalize();

  return retval;
}

//
// Test the LCM ROL mini minimizer.
//
TEST(AlbanyResidualROL, LineSearchRosenbrock)
{
  bool const
  print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream
  bhs;

  std::ostream &
  os = (print_output == true) ? std::cout : bhs;

  using EvalT = PHAL::AlbanyTraits::Residual;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  constexpr
  minitensor::Index
  DIM{2};

  using FN = LCM::Banana_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;

  ValueT const
  a = 1.0;

  ValueT const
  b = 100.0;

  FN
  fn(a, b);

  MIN
  minimizer;

  // Define algorithm.
  std::string const
  algoname{"Line Search"};

  // Set parameters.
  Teuchos::ParameterList
  params;

  params.sublist("Step").sublist("Line Search").sublist("Descent Method").
    set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolverROL<MIN, FN, EvalT, DIM>
  mini_solver(minimizer, algoname, params, fn, x);

  minimizer.printReport(os);

  ASSERT_EQ(minimizer.converged, true);
}

TEST(AlbanyJacobianROL, LineSearchRosenbrock)
{
  bool const
  print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream
  bhs;

  std::ostream &
  os = (print_output == true) ? std::cout : bhs;

  using EvalT = PHAL::AlbanyTraits::Jacobian;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  constexpr
  minitensor::Index
  DIM{2};

  using FN = LCM::Banana_Traits<EvalT>;
  using MIN = ROL::MiniTensor_Minimizer<ValueT, DIM>;

  ValueT const
  a = 1.0;

  ValueT const
  b = 100.0;

  FN
  fn(a, b);

  MIN
  minimizer;

  // Define algorithm.
  std::string const
  algoname{"Line Search"};

  // Set parameters.
  Teuchos::ParameterList
  params;

  params.sublist("Step").sublist("Line Search").sublist("Descent Method").
    set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  minitensor::Vector<ScalarT, DIM>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolverROL<MIN, FN, EvalT, DIM>
  mini_solver(minimizer, algoname, params, fn, x);

  minimizer.printReport(os);

  ASSERT_EQ(minimizer.converged, true);
}
