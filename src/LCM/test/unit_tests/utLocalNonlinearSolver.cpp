//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <LocalNonlinearSolver.hpp>
#include <Sacado.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "PHAL_AlbanyTraits.hpp"

using namespace std;

namespace {

TEUCHOS_UNIT_TEST(LocalNonlinearSolver, Instantiation)
{
  typedef PHAL::AlbanyTraits                    Traits;
  typedef PHAL::AlbanyTraits::Residual          EvalT;
  typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  int numLocalVars(2);

  std::vector<ScalarT>                     F(numLocalVars);
  std::vector<ScalarT>                     dFdX(numLocalVars * numLocalVars);
  std::vector<ScalarT>                     X(numLocalVars);
  LCM::LocalNonlinearSolver<EvalT, Traits> solver;

  const int n      = 2;
  const int nrhs   = 1;
  RealType  A[]    = {1.1, 0.1, .01, 0.9};
  const int lda    = 2;
  int       IPIV[] = {0, 0};
  RealType  B[]    = {0.1, 0.2};
  const int ldb    = 2;
  int       info(0);

  const RealType refX[] = {0.088978766430738, 0.212335692618807};

  // this is simply testing if we can call lapack through the interface
  solver.lapack.GESV(n, nrhs, &A[0], lda, &IPIV[0], &B[0], ldb, &info);

  TEST_COMPARE(fabs(B[0] - refX[0]), <=, 1.0e-15);
  TEST_COMPARE(fabs(B[1] - refX[1]), <=, 1.0e-15);
}

TEUCHOS_UNIT_TEST(LocalNonlinearSolver, Residual)
{
  typedef PHAL::AlbanyTraits                    Traits;
  typedef PHAL::AlbanyTraits::Residual          EvalT;
  typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  // local objective function and solution
  int                                      numLocalVars(1);
  std::vector<ScalarT>                     F(numLocalVars);
  std::vector<ScalarT>                     dFdX(numLocalVars * numLocalVars);
  std::vector<ScalarT>                     X(numLocalVars);
  LCM::LocalNonlinearSolver<EvalT, Traits> solver;

  // initialize X
  X[0] = 1.0;

  int  count(0);
  bool converged = false;
  while (!converged && count < 10) {
    // objective function --> x^2 - 2 == 0
    F[0]    = X[0] * X[0] - 2.0;
    dFdX[0] = 2.0 * X[0];

    solver.solve(dFdX, X, F);

    if (fabs(F[0]) <= 1.0E-15) converged = true;

    count++;
  }
  F[0] = X[0] * X[0] - 2.0;
  std::vector<ScalarT> sol(numLocalVars);
  solver.computeFadInfo(dFdX, X, F);

  const RealType refX[] = {std::sqrt(2)};
  TEST_COMPARE(fabs(X[0] - refX[0]), <=, 1.0e-15);
}

TEUCHOS_UNIT_TEST(LocalNonlinearSolver, Jacobian)
{
  typedef PHAL::AlbanyTraits                    Traits;
  typedef PHAL::AlbanyTraits::Jacobian          EvalT;
  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

  // local objective function and solution
  int                                      numLocalVars(1);
  std::vector<ScalarT>                     F(numLocalVars);
  std::vector<ScalarT>                     dFdX(numLocalVars * numLocalVars);
  std::vector<ScalarT>                     X(numLocalVars);
  LCM::LocalNonlinearSolver<EvalT, Traits> solver;

  // initialize X
  X[0] = 1.0;

  ScalarT two(1, 0, 2.0);
  int     count(0);
  bool    converged = false;
  while (!converged && count < 10) {
    // objective function --> x^2 - 2 == 0
    F[0]    = X[0] * X[0] - two;
    dFdX[0] = 2.0 * X[0];

    solver.solve(dFdX, X, F);

    if (fabs(F[0]) <= 1.0E-15) converged = true;

    count++;
  }

  F[0] = X[0] * X[0] - two;
  solver.computeFadInfo(dFdX, X, F);

  const RealType refX[] = {std::sqrt(2)};
  TEST_COMPARE(fabs(X[0].val() - refX[0]), <=, 1.0e-15);
}
TEUCHOS_UNIT_TEST(LocalNonlinearSolver, Tangent)
{
  typedef PHAL::AlbanyTraits                   Traits;
  typedef PHAL::AlbanyTraits::Tangent          EvalT;
  typedef PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;

  // local objective function and solution
  int                                      numLocalVars(1);
  std::vector<ScalarT>                     F(numLocalVars);
  std::vector<ScalarT>                     dFdX(numLocalVars * numLocalVars);
  std::vector<ScalarT>                     X(numLocalVars);
  LCM::LocalNonlinearSolver<EvalT, Traits> solver;

  // initialize X
  X[0] = 1.0;

  ScalarT two(1, 0, 2.0);

  int  count(0);
  bool converged = false;
  while (!converged && count < 10) {
    // objective function --> x^2 - 2 == 0
    F[0]    = X[0] * X[0] - two;
    dFdX[0] = 2.0 * X[0];

    solver.solve(dFdX, X, F);

    if (fabs(F[0]) <= 1.0E-15) converged = true;

    count++;
  }

  F[0] = X[0] * X[0] - two;
  solver.computeFadInfo(dFdX, X, F);

  const RealType refX[] = {std::sqrt(2)};
  TEST_COMPARE(fabs(X[0].val() - refX[0]), <=, 1.0e-15);
}
}  // namespace
