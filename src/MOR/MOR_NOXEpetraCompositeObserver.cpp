//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_NOXEpetraCompositeObserver.hpp"

namespace MOR {

void
NOXEpetraCompositeObserver::observeSolution(const Epetra_Vector& solution)
{
  for (ObserverSequence::iterator it = observers_.begin(), it_end = observers_.end(); it != it_end; ++it) {
    (*it)->observeSolution(solution);
  }
}

void
NOXEpetraCompositeObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  for (ObserverSequence::iterator it = observers_.begin(), it_end = observers_.end(); it != it_end; ++it) {
    (*it)->observeSolution(solution, time_or_param_val);
  }
}

int
NOXEpetraCompositeObserver::observerCount() const
{
  return observers_.size();
}

void
NOXEpetraCompositeObserver::addObserver(const Teuchos::RCP<NOX::Epetra::Observer> &obs)
{
  observers_.push_back(obs);
}

} // end namespace MOR
