//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
