//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "NOX_Abstract_Group.H"
#include "NOX_Solver_Generic.H"
#include "NOXSolverPrePostOperator.h"

//
//
//
NOXSolverPrePostOperator::
NOXSolverPrePostOperator()
{
  return;
}

//
//
//
NOXSolverPrePostOperator::
~NOXSolverPrePostOperator()
{
  return;
}

//
//
//
void
NOXSolverPrePostOperator::
runPreIterate(NOX::Solver::Generic const &)
{
  return;
}

//
//
//
void
NOXSolverPrePostOperator::
runPostIterate(NOX::Solver::Generic const &)
{
  return;
}

//
//
//
void
NOXSolverPrePostOperator::
runPreSolve(NOX::Solver::Generic const & solver)
{
  // This is needed for step reduction if numerics fails.
  if(status_test_.is_null() == false){
    status_test_->status_ = NOX::StatusTest::Unevaluated;
  }

  // This is needed for Schwarz coupling
  NOX::Abstract::Vector const &
  x = solver.getSolutionGroup().getX();

  norm_init_ = x.norm();

  soln_init_ = x.clone();

  return;
}

//
//
//
void
NOXSolverPrePostOperator::
runPostSolve(NOX::Solver::Generic const & solver)
{
  NOX::Abstract::Vector const &
  y = solver.getSolutionGroup().getX();

  norm_final_ = y.norm();

  NOX::Abstract::Vector const &
  x = *(soln_init_);

  Teuchos::RCP<NOX::Abstract::Vector>
  soln_diff = x.clone();

  NOX::Abstract::Vector &
  dx = *(soln_diff);

  dx.update(1.0, y, -1.0, x, 0.0);

  norm_diff_ = dx.norm();

  return;
}

//
//
//
void
NOXSolverPrePostOperator::
setStatusTest(Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test)
{
  status_test_ = status_test;
}
//
//
//
ST
NOXSolverPrePostOperator::
getInitialNorm()
{
  return norm_init_;
}

//
//
//
ST
NOXSolverPrePostOperator::
getFinalNorm()
{
  return norm_final_;
}

//
//
//
ST
NOXSolverPrePostOperator::
getDifferenceNorm()
{
  return norm_diff_;
}
