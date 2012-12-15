//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProjectionErrorObserver.hpp"

namespace Albany {

ProjectionErrorObserver::ProjectionErrorObserver(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile,
      const Teuchos::RCP<NOX::Epetra::Observer>& decoratedObserver):
  projectionError_(projectionSpace, errorFile),
  decoratedObserver_(decoratedObserver)
{
   // Nothing to do
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution)
{
  decoratedObserver_->observeSolution(solution);
  projectionError_.process(solution);
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  decoratedObserver_->observeSolution(solution, time_or_param_val);
  projectionError_.process(solution);
}

} // end namespace Albany
