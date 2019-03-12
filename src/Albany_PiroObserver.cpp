//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PiroObserver.hpp"
#include "Albany_ObserverImpl.hpp"

namespace Albany
{

PiroObserver::
PiroObserver(const Teuchos::RCP<Albany::Application>& app)
 : impl_(new ObserverImpl(app))
{
  // Nothing to be done here
}

void PiroObserver::observeSolution(const Thyra_Vector& solution)
{
  // Determine the stamp associated with the snapshot
  const double stamp = impl_->getTimeParamValueOrDefault(0.0);

  impl_->observeSolution(stamp, solution, Teuchos::null, Teuchos::null);
} 

} // namespace Albany
