//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_SnapshotCollectionObserver.hpp"

namespace MOR {

SnapshotCollectionObserver::SnapshotCollectionObserver(
    int period,
    const Teuchos::RCP<MultiVectorOutputFile> &snapshotFile,
    const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver) :
  snapshotCollector_(period, snapshotFile),
  decoratedObserver_(decoratedObserver)
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

} // namespace MOR
