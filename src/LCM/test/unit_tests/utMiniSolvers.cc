//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <gtest/gtest.h>
#include <MiniLinearSolver.h>
#include <MiniNonlinearSolver.h>
#include <MiniSolvers.h>

// Why is this needed?
bool TpetraBuild = false;

namespace
{
//
// Test the LCM mini minimizer.
//
TEST(AlbanyResidual, NewtonBanana)
{
  using EvalT = PHAL::AlbanyTraits::Residual;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  constexpr
  Intrepid2::Index
  dim{2};

  using MIN = Intrepid2::Minimizer<ValueT, dim>;
  using FN = LCM::BananaNLS<ValueT>;
  using STEP = Intrepid2::StepBase<FN, ValueT, dim>;

  MIN
  minimizer;

  std::unique_ptr<STEP>
  pstep =
      Intrepid2::stepFactory<FN, ValueT, dim>(Intrepid2::StepType::NEWTON);

  assert(pstep->name() != nullptr);

  STEP &
  step = *pstep;

  FN
  banana;

  Intrepid2::Vector<ScalarT, dim>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolver<MIN, STEP, FN, EvalT, dim>
  mini_solver(minimizer, step, banana, x);

  minimizer.printReport(std::cout);

  ASSERT_EQ(minimizer.converged, true);
}

//
// Test the LCM mini minimizer.
//
TEST(AlbanyJacobian, NewtonBanana)
{
  using EvalT = PHAL::AlbanyTraits::Jacobian;
  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  constexpr
  Intrepid2::Index
  dim{2};

  using MIN = Intrepid2::Minimizer<ValueT, dim>;
  using FN = LCM::BananaNLS<ValueT>;
  using STEP = Intrepid2::NewtonStep<FN, ValueT, dim>;

  MIN
  minimizer;

  STEP
  step;

  FN
  banana;

  Intrepid2::Vector<ScalarT, dim>
  x;

  x(0) = 0.0;
  x(1) = 3.0;

  LCM::MiniSolver<MIN, STEP, FN, EvalT, dim>
  mini_solver(minimizer, step, banana, x);

  minimizer.printReport(std::cout);

  ASSERT_EQ(minimizer.converged, true);
}

} // anonymous namespace

int
main(int ac, char * av[])
{
  Kokkos::initialize();

  ::testing::InitGoogleTest(&ac, av);

  return RUN_ALL_TESTS();

  Kokkos::finalize();
}
