//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "NOX_StatusTest_ModelEvaluatorFlag.h"
#include <vector>
#include "Albany_Utils.hpp"

NOX::StatusTest::ModelEvaluatorFlag::ModelEvaluatorFlag() : status_(Unevaluated)
{
}

NOX::StatusTest::ModelEvaluatorFlag::~ModelEvaluatorFlag() {}

NOX::StatusTest::StatusType
NOX::StatusTest::ModelEvaluatorFlag::checkStatus(
    const Solver::Generic&     problem,
    NOX::StatusTest::CheckType checkType)
{
  syncFlag();

  switch (checkType) {
    case NOX::StatusTest::Complete:
    case NOX::StatusTest::Minimal:

      // In these cases we'll return status_ as set by any ModelEvaluator
      // that has access to the StatusTest object.
      // If the ModelEvaluators take no action, then status_ will be
      // Unevaluated, in which case Unconverged will be returned.
      // If a ModelEveluator wants to trigger a NOX failure,
      // the ModelEvaluator will have set status_ to Failed.

      if (status_ == Unevaluated) { status_ = Unconverged; }

      break;

    case NOX::StatusTest::None:
    default: status_ = Unevaluated; break;
  }

  return status_;
}

NOX::StatusTest::StatusType
NOX::StatusTest::ModelEvaluatorFlag::getStatus() const
{
  return status_;
}

void
NOX::StatusTest::ModelEvaluatorFlag::syncFlag()
{
  std::vector<int> localVal(1), globalVal(1);
  if (status_ == NOX::StatusTest::Unevaluated) {
    localVal[0] = 0;
  } else if (status_ == NOX::StatusTest::Converged) {
    localVal[0] = 1;
  } else if (status_ == NOX::StatusTest::Unconverged) {
    localVal[0] = 2;
  } else if (status_ == NOX::StatusTest::Failed) {
    localVal[0] = 3;
  }

  Teuchos::RCP<Teuchos_Comm> teuchosComm =
      Albany::createTeuchosCommFromMpiComm(Albany_MPI_COMM_WORLD);
  Teuchos::reduceAll(
      *teuchosComm, Teuchos::REDUCE_MAX, 1, &localVal[0], &globalVal[0]);

  if (globalVal[0] == 0) {
    status_ = NOX::StatusTest::Unevaluated;
  } else if (globalVal[0] == 1) {
    status_ = NOX::StatusTest::Converged;
  } else if (globalVal[0] == 2) {
    status_ = NOX::StatusTest::Unconverged;
  } else if (globalVal[0] == 3) {
    status_ = NOX::StatusTest::Failed;
  }
}

std::ostream&
NOX::StatusTest::ModelEvaluatorFlag::print(std::ostream& stream, int indent)
    const
{
  for (int j = 0; j < indent; j++) { stream << ' '; }
  stream << status_;
  stream << "Model Evaluator Flag: ";
  stream << status_message_;
  stream << std::endl;

  return stream;
}
