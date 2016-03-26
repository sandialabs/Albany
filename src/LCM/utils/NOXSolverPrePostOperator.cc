//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "NOXSolverPrePostOperator.h"
#include <iostream>

void NOXSolverPrePostOperator::runPreSolve(const NOX::Solver::Generic& solver)
{
  if(!status_test_.is_null()){
    status_test_->status_ = NOX::StatusTest::Unevaluated;
  }
}
