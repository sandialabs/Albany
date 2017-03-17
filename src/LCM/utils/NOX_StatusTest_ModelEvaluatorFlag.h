//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef NOX_STATUS_MODELEVALUATORFLAG_H
#define NOX_STATUS_MODELEVALUATORFLAG_H

#include "NOX_StatusTest_Generic.H"  // base class

namespace NOX {

namespace StatusTest {

//! Failure test based on a flag set by a ModelEvaluator.
class ModelEvaluatorFlag : public Generic {

public:

  //! Constructor.
    ModelEvaluatorFlag();

  //! Destructor.
  virtual ~ModelEvaluatorFlag();

  virtual NOX::StatusTest::StatusType
  checkStatus(const NOX::Solver::Generic& problem,
	      NOX::StatusTest::CheckType checkType);

  virtual NOX::StatusTest::StatusType getStatus() const;

  void syncFlag();

  virtual std::ostream& print(std::ostream& stream, int indent = 0) const;

  //! Current status
  NOX::StatusTest::StatusType status_;

  std::string
  status_message_;
};

} // namespace StatusTest
} // namespace NOX

#endif
