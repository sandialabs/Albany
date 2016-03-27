//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "NOX_StatusTest_ModelEvaluatorFlag.h"

NOX::StatusTest::ModelEvaluatorFlag::
ModelEvaluatorFlag() :
  status_(Unevaluated)
{
}

NOX::StatusTest::ModelEvaluatorFlag::~ModelEvaluatorFlag()
{
}

NOX::StatusTest::StatusType NOX::StatusTest::ModelEvaluatorFlag::
checkStatus(const Solver::Generic& problem,
	    NOX::StatusTest::CheckType checkType)
{
  switch (checkType)
  {
  case NOX::StatusTest::Complete:
  case NOX::StatusTest::Minimal:

    // In these cases we'll return status_ as set by any ModelEvaluator
    // that has access to the StatusTest object.
    // If the ModelEvaluators take no action, then status_ will be
    // Unevaluated, in which case Unconverged will be returned.
    // If a ModelEveluator wants to trigger a NOX failure,
    // the ModelEvaluator will have set status_ to Failed.

    if (status_ == Unevaluated){
      status_ = Unconverged;
    }

    break;

  case NOX::StatusTest::None:
  default:

    status_ = Unevaluated;
    break;
  }

  return status_;
}

NOX::StatusTest::StatusType NOX::StatusTest::ModelEvaluatorFlag::getStatus() const
{
  return status_;
}

std::ostream& NOX::StatusTest::ModelEvaluatorFlag::print(std::ostream& stream, int indent) const
{
  for (int j = 0; j < indent; j ++){
    stream << ' ';
  }
  stream << status_;
  stream << "Model Evaluator Flag";
  stream << std::endl;

  return stream;
}
