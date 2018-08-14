//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MiniLinearSolver.h"
#include "MiniNonlinearSolver.h"
#include "MiniSolvers.h"
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
// Test the LCM mini minimizer.
//
TEST(AlbanyResidual, NewtonBanana)
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  using EvalT   = PHAL::AlbanyTraits::Residual;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using MIN  = minitensor::Minimizer<ValueT, DIM>;
  using FN   = LCM::Banana<ValueT>;
  using STEP = minitensor::StepBase<FN, ValueT, DIM>;

  MIN minimizer;

  std::unique_ptr<STEP> pstep =
      minitensor::stepFactory<FN, ValueT, DIM>(minitensor::StepType::NEWTON);

  assert(pstep->name() != nullptr);

  STEP& step = *pstep;

  FN banana;

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolver<MIN, STEP, FN, EvalT, DIM> mini_solver(
      minimizer, step, banana, x);

  minimizer.printReport(os);

  ASSERT_EQ(minimizer.converged, true);
}

//
// Test the LCM mini minimizer.
//
TEST(AlbanyJacobian, NewtonBanana)
{
  bool const print_output = ::testing::GTEST_FLAG(print_time);

  // outputs nothing
  Teuchos::oblackholestream bhs;

  std::ostream& os = (print_output == true) ? std::cout : bhs;

  using EvalT   = PHAL::AlbanyTraits::Jacobian;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT  = typename Sacado::ValueType<ScalarT>::type;

  constexpr minitensor::Index DIM{2};

  using MIN  = minitensor::Minimizer<ValueT, DIM>;
  using FN   = LCM::Banana<ValueT>;
  using STEP = minitensor::NewtonStep<FN, ValueT, DIM>;

  MIN minimizer;

  STEP step;

  FN banana;

  minitensor::Vector<ScalarT, DIM> x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolver<MIN, STEP, FN, EvalT, DIM> mini_solver(
      minimizer, step, banana, x);

  minimizer.printReport(os);

  ASSERT_EQ(minimizer.converged, true);
}
