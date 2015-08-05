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
