//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "SolutionSniffer.hpp"
#include "NOX_Abstract_Group.H"
#include "NOX_Solver_Generic.H"
#include "Teuchos_VerboseObject.hpp"

//#define DEBUG

namespace LCM {

//
//
//
SolutionSniffer::SolutionSniffer() { return; }

//
//
//
SolutionSniffer::~SolutionSniffer() { return; }

//
//
//
void
SolutionSniffer::runPreIterate(NOX::Solver::Generic const&)
{
  return;
}

//
//
//
void
SolutionSniffer::runPostIterate(NOX::Solver::Generic const&)
{
  return;
}

//
//
//
void
SolutionSniffer::runPreSolve(NOX::Solver::Generic const& solver)
{
  if (status_test_.is_null() == false) {
    status_test_->status_         = NOX::StatusTest::Unevaluated;
    status_test_->status_message_ = "";
  }

  NOX::Abstract::Vector const& x = solver.getPreviousSolutionGroup().getX();

  norm_init_ = x.norm();

  soln_init_ = x.clone(NOX::DeepCopy);

#if defined(DEBUG)
  Teuchos::FancyOStream& fos = *Teuchos::VerboseObjectBase::getDefaultOStream();

  fos << "\n*** NOX: Initial solution ***\n";
  x.print(fos);
  fos << "\n*** NOX: Initial solution ***\n";
#endif  // DEBUG

  return;
}

//
//
//
void
SolutionSniffer::runPostSolve(NOX::Solver::Generic const& solver)
{
  NOX::Abstract::Vector const& y = solver.getSolutionGroup().getX();

  // Save solution
  last_soln_ = y.clone();

  norm_final_ = y.norm();

  NOX::Abstract::Vector const& x = *(soln_init_);

  Teuchos::RCP<NOX::Abstract::Vector> soln_diff = x.clone();

  NOX::Abstract::Vector& dx = *(soln_diff);

  dx.update(1.0, y, -1.0, x, 0.0);

  norm_diff_ = dx.norm();

#if defined(DEBUG)
  Teuchos::FancyOStream& fos = *Teuchos::VerboseObjectBase::getDefaultOStream();

  fos << "\n*** NOX: Final solution ***\n";
  y.print(fos);
  fos << "\n*** NOX: Final solution ***\n";
#endif

  return;
}

//
//
//
void
SolutionSniffer::setStatusTest(
    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> status_test)
{
  status_test_ = status_test;
}

//
//
//
ST
SolutionSniffer::getInitialNorm()
{
  return norm_init_;
}

//
//
//
ST
SolutionSniffer::getFinalNorm()
{
  return norm_final_;
}

//
//
//
ST
SolutionSniffer::getDifferenceNorm()
{
  return norm_diff_;
}

//
//
//
Teuchos::RCP<NOX::Abstract::Vector>
SolutionSniffer::getLastSoln()
{
  return last_soln_;
}

}  // namespace LCM
