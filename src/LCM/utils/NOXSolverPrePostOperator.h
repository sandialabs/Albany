//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(NOXSOLVERPREPOSTOPERATOR_H)
#define NOXSOLVERPREPOSTOPERATOR_H

#include "Albany_DataTypes.hpp"
#include "NOX_Abstract_PrePostOperator.H"
#include "NOX_Abstract_Vector.H"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"

namespace LCM {

///
/// Observer that is called at various points in the NOX nonlinear solver
///
class NOXSolverPrePostOperator : public NOX::Abstract::PrePostOperator {

public:

  /// Constructor.
  NOXSolverPrePostOperator();

  /// Destructor.
  virtual
  ~NOXSolverPrePostOperator();

  /// Set the status test
  void
  setStatusTest(Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test);

  virtual void
  runPreIterate(NOX::Solver::Generic const & solver);

  virtual void
  runPostIterate(NOX::Solver::Generic const & solver);

  virtual void
  runPreSolve(NOX::Solver::Generic const & solver);

  virtual void
  runPostSolve(NOX::Solver::Generic const & solver);

  ST
  getInitialNorm();

  ST
  getFinalNorm();

  ST
  getDifferenceNorm();

private:

  // For step reduction
  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>
  status_test_{Teuchos::null};

  // For Schwarz coupling
  Teuchos::RCP<NOX::Abstract::Vector>
  soln_init_{Teuchos::null};

  ST
  norm_init_{3.0};

  ST
  norm_final_{5.0};

  ST
  norm_diff_{7.0};
};

} // namespace LCM

#endif // NOXSOLVERPREPOSTOPERATOR_H
