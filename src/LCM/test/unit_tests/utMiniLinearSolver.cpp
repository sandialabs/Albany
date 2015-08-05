//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_UnitTestHarness.hpp>
#include <MiniLinearSolver.hpp>
#include <Sacado.hpp>
#include "PHAL_AlbanyTraits.hpp"

namespace
{

TEUCHOS_UNIT_TEST(MiniLinearSolver, Instantiation)
{
  using Traits = PHAL::AlbanyTraits;
  using EvalT = PHAL::AlbanyTraits::Residual;
  using ScalarT =  PHAL::AlbanyTraits::Residual::ScalarT;

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

  Intrepid::Vector<RealType, dimension> const
  x;

  LCM::MiniLinearSolver<EvalT, Traits>
  solver;

  solver.solve(A, b, x);

  RealType const
  error = norm(x - v) / norm(v);

  TEST_COMPARE(error, <=, Intrepid::machine_epsilon<RealType>());
}

TEUCHOS_UNIT_TEST(MiniLinearSolver, Residual)
{
  using Traits = PHAL::AlbanyTraits;
  using EvalT = PHAL::AlbanyTraits::Residual;
  using ScalarT =  PHAL::AlbanyTraits::Residual::ScalarT;
}

TEUCHOS_UNIT_TEST(MiniLinearSolver, Jacobian)
{
  using Traits = PHAL::AlbanyTraits;
  using EvalT = PHAL::AlbanyTraits::Residual;
  using ScalarT =  PHAL::AlbanyTraits::Residual::ScalarT;
}

TEUCHOS_UNIT_TEST(MiniLinearSolver, Tangent)
{
  using Traits = PHAL::AlbanyTraits;
  using EvalT = PHAL::AlbanyTraits::Residual;
  using ScalarT =  PHAL::AlbanyTraits::Residual::ScalarT;
}

} // anonymous namespace
