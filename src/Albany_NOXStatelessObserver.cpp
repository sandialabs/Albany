//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_NOXStatelessObserver.hpp"

#include "Albany_StatelessObserverImpl.hpp"
#include "Albany_EpetraThyraUtils.hpp"

namespace Albany
{

NOXStatelessObserver::
NOXStatelessObserver (const Teuchos::RCP<Albany::Application> &app)
  : impl(new StatelessObserverImpl(app))
{
  // Nothing to be done here
}

void NOXStatelessObserver::observeSolution (const Epetra_Vector& solution) {
  observeSolution(solution, impl->getTimeParamValueOrDefault(0.0));
}

void NOXStatelessObserver::observeSolution (const Epetra_Vector& solution,
                                            double time_or_param_val) {
  Teuchos::RCP<const Epetra_Vector> solution_ptr = Teuchos::rcpFromRef(solution);
  auto solution_thyra = createConstThyraVector(solution_ptr);
  impl->observeSolution(time_or_param_val, *solution_thyra, Teuchos::null);
}

} // namespace Albany
