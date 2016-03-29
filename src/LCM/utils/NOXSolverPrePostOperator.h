//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef NOXSOLVERPREPOSTOPERATOR_H
#define NOXSOLVERPREPOSTOPERATOR_H

#include <NOX_Abstract_PrePostOperator.H>
#include <Teuchos_RCP.hpp>
#include "NOX_StatusTest_ModelEvaluatorFlag.h"

//! Observer that is called at various point in the NOX nonlinear solve process
class NOXSolverPrePostOperator : public NOX::Abstract::PrePostOperator {

public:

  //! Constructor.
  NOXSolverPrePostOperator() {}

  //! Destructor.
  virtual ~NOXSolverPrePostOperator() {}

  //! Set the status test
  void setStatusTest(Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test) { status_test_ = status_test; }

  //! User defined method that will be executed at the start of a call to NOX::Solver::Generic::iterate().
  //virtual void runPreIterate(const NOX::Solver::Generic& solver);

  //! User defined method that will be executed at the end of a call to NOX::Solver::Generic::iterate().
  //virtual void runPostIterate(const NOX::Solver::Generic& solver);

  //! User defined method that will be executed at the start of a call to NOX::Solver::Generic::solve().
  virtual void runPreSolve(const NOX::Solver::Generic& solver);

  //! User defined method that will be executed at the end of a call to NOX::Solver::Generic::solve().
  //virtual void runPostSolve(const NOX::Solver::Generic& solver);

protected:

  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test_;
};

#endif
