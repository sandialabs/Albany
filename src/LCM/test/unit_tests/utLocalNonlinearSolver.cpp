/********************************************************************  \
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include <Teuchos_UnitTestHarness.hpp>
#include <LocalNonlinearSolver.h>
#include <Sacado.hpp>
#include "PHAL_AlbanyTraits.hpp"
#include "Tensor.h"

using namespace std;

namespace {

  TEUCHOS_UNIT_TEST( LocalNonlinearSolver, Instantiation )
  {
    typedef PHAL::AlbanyTraits::Residual EvalT;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

    int numLocalVars(2);

    std::vector<ScalarT> F(numLocalVars);
    std::vector<ScalarT> dFdX(numLocalVars*numLocalVars);
    std::vector<ScalarT> X(numLocalVars);
    LCM::LocalNonlinearSolver<EvalT> solver;

    const int n = 2;
    const int nrhs = 1;
    RealType A[] = { 1.1, 0.1, .01, 0.9 };
    const int lda = 2;
    int IPIV[] = {0, 0};
    RealType B[] = { 0.1, 0.2 };
    const int ldb = 2;
    int info(0);

    const RealType refX[] = {0.088978766430738, 0.212335692618807};

    // this is simply testing if we can call lapack through the interface
    solver.lapack.GESV(n, nrhs, &A[0], lda, &IPIV[0], &B[0], ldb, &info);
 
    TEST_COMPARE( fabs(B[0]-refX[0]), <=, 1.0e-15 );
    TEST_COMPARE( fabs(B[1]-refX[1]), <=, 1.0e-15 );
  }

  TEUCHOS_UNIT_TEST( LocalNonlinearSolver, Residual )
  {
    typedef PHAL::AlbanyTraits::Residual EvalT;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

    // local objective function and solution
    int numLocalVars(1);
    std::vector<ScalarT> F(numLocalVars);
    std::vector<ScalarT> dFdX(numLocalVars*numLocalVars);
    std::vector<ScalarT> X(numLocalVars);
    LCM::LocalNonlinearSolver<EvalT> solver;
    
    // initialize X
    X[0] = 1.0;
    
    int count(0);
    bool converged = false;
    while ( !converged && count < 10 )
    {
      // objective function --> x^2 - 2 == 0
      F[0] = X[0]*X[0] - 2.0;
      dFdX[0] = 2.0 * X[0];

      solver.solve(dFdX,X,F);

      if (fabs(F[0]) <= 1.0E-15 )
        converged = true;

      count++;
    }
    F[0] = X[0]*X[0] - 2.0;
    std::vector<ScalarT> sol(numLocalVars);
    solver.computeFadInfo(dFdX,X,F);

    const RealType refX[] = { std::sqrt(2) }; 
    TEST_COMPARE( fabs(X[0]-refX[0]), <=, 1.0e-15 );

  }

  TEUCHOS_UNIT_TEST( LocalNonlinearSolver, Jacobian )
  {
    typedef PHAL::AlbanyTraits::Jacobian EvalT;
    typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

    // local objective function and solution
    int numLocalVars(1);
    std::vector<ScalarT> F(numLocalVars);
    std::vector<ScalarT> dFdX(numLocalVars*numLocalVars);
    std::vector<ScalarT> X(numLocalVars);
    LCM::LocalNonlinearSolver<EvalT> solver;
    
    // initialize X
    X[0] = 1.0;
    
    ScalarT two(1,0,2.0);
    int count(0);
    bool converged = false;
    while ( !converged && count < 10 )
    {
      // objective function --> x^2 - 2 == 0
      F[0] = X[0]*X[0] - two;
      dFdX[0] = 2.0 * X[0];

      solver.solve(dFdX,X,F);

      if (fabs(F[0]) <= 1.0E-15 )
        converged = true;

      count++;
    }

    F[0] = X[0]*X[0] - two;
    solver.computeFadInfo(dFdX,X,F);

    const RealType refX[] = { std::sqrt(2) }; 
    TEST_COMPARE( fabs(X[0].val()-refX[0]), <=, 1.0e-15 );

  }
  TEUCHOS_UNIT_TEST( LocalNonlinearSolver, Tangent )
  {
    typedef PHAL::AlbanyTraits::Tangent EvalT;
    typedef PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;

    // local objective function and solution
    int numLocalVars(1);
    std::vector<ScalarT> F(numLocalVars);
    std::vector<ScalarT> dFdX(numLocalVars*numLocalVars);
    std::vector<ScalarT> X(numLocalVars);
    LCM::LocalNonlinearSolver<EvalT> solver;
    
    // initialize X
    X[0] = 1.0;
    
    ScalarT two(1,0,2.0);

    int count(0);
    bool converged = false;
    while ( !converged && count < 10 )
    {
      // objective function --> x^2 - 2 == 0
      F[0] = X[0]*X[0] - two;
      dFdX[0] = 2.0 * X[0];

      solver.solve(dFdX,X,F);

      if (fabs(F[0]) <= 1.0E-15 )
        converged = true;

      count++;
    }

    F[0] = X[0]*X[0] - two;
    solver.computeFadInfo(dFdX,X,F);

    const RealType refX[] = { std::sqrt(2) }; 
    TEST_COMPARE( fabs(X[0].val()-refX[0]), <=, 1.0e-15 );
  }
} // namespace
