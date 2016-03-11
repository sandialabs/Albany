//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: right now this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#include "Albany_NOXObserver.hpp"

#include "Teuchos_ENull.hpp"

Albany_NOXObserver::Albany_NOXObserver(
    const Teuchos::RCP<Albany::Application> &app_) :
  impl(app_)
{
   // Nothing to do
}

void Albany_NOXObserver::observeSolution(
    const Epetra_Vector& solution)
{
  this->observeSolution(solution, impl.getTimeParamValueOrDefault(0.0));
}

void Albany_NOXObserver::observeSolution(
    const Epetra_Vector& solution, double time_or_param_val)
{
  impl.observeSolution(time_or_param_val, solution, Teuchos::null);
}
