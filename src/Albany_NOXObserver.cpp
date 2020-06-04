//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_NOXObserver.hpp"
#include "Albany_ObserverImpl.hpp"
#include "Albany_EpetraThyraUtils.hpp"

namespace Albany
{

NOXObserver::NOXObserver(const Teuchos::RCP<Albany::Application>& app_)
 : impl(new ObserverImpl(app_))
{
   // Nothing to do
}

void NOXObserver::observeSolution(const Epetra_Vector& solution)
{
  this->observeSolution(solution, impl->getTimeParamValueOrDefault(0.0));
}

void NOXObserver::observeSolution(const Epetra_Vector& solution,
                                  double time_or_param_val)
{
  auto solution_ptr = Teuchos::rcpFromRef(solution);
  auto solution_thyra = createConstThyraVector(solution_ptr);
  impl->observeSolution(time_or_param_val, *solution_thyra, Teuchos::null, Teuchos::null, Teuchos::null);
}

} // namespace Albany
