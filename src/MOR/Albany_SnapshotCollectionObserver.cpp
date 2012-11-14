//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_SnapshotCollectionObserver.hpp"

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;

SnapshotCollectionObserver::SnapshotCollectionObserver(const RCP<ParameterList> &params,
                                                       const Teuchos::RCP<NOX::Epetra::Observer>& decoratedObserver) :
  decoratedObserver_(decoratedObserver),
  snapshotCollector_(params)
{
   // Nothing to do
}

void SnapshotCollectionObserver::observeSolution(const Epetra_Vector& solution)
{
  decoratedObserver_->observeSolution(solution);
  snapshotCollector_.addVector(0.0, solution);
}

void SnapshotCollectionObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  decoratedObserver_->observeSolution(solution, time_or_param_val);
  snapshotCollector_.addVector(time_or_param_val, solution);
}

} // end namespace Albany
