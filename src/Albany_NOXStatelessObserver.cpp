//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_NOXStatelessObserver.hpp"
#include "Teuchos_ENull.hpp"

Albany::NOXStatelessObserver::
NOXStatelessObserver (const Teuchos::RCP<Albany::Application> &app)
  : impl(app)
{}

void Albany::NOXStatelessObserver::
observeSolution (const Epetra_Vector& solution) {
  observeSolution(solution, impl.getTimeParamValueOrDefault(0.0));
}

void Albany::NOXStatelessObserver::
observeSolution (const Epetra_Vector& solution, double time_or_param_val) {
  impl.observeSolution(time_or_param_val, solution, Teuchos::null);
}
