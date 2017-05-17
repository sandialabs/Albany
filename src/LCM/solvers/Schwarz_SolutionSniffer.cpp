//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "NOX_Abstract_Group.H"
#include "NOX_Solver_Generic.H"
#include "Schwarz_SolutionSniffer.hpp"

namespace LCM {

//
//
//
SolutionSniffer::
SolutionSniffer()
{
  return;
}

//
//
//
SolutionSniffer::
~SolutionSniffer()
{
  return;
}

//
//
//
void
SolutionSniffer::
runPreIterate(NOX::Solver::Generic const &)
{
  return;
}

//
//
//
void
SolutionSniffer::
runPostIterate(NOX::Solver::Generic const &)
{
  return;
}

//
//
//
void
SolutionSniffer::
runPreSolve(NOX::Solver::Generic const & solver)
{
  NOX::Abstract::Vector const &
  x = solver.getPreviousSolutionGroup().getX();

  norm_init_ = x.norm();

  soln_init_ = x.clone(NOX::DeepCopy);

  return;
}

//
//
//
void
SolutionSniffer::
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
ST
SolutionSniffer::
getInitialNorm()
{
  return norm_init_;
}

//
//
//
ST
SolutionSniffer::
getFinalNorm()
{
  return norm_final_;
}

//
//
//
ST
SolutionSniffer::
getDifferenceNorm()
{
  return norm_diff_;
}

} // namespace LCM
