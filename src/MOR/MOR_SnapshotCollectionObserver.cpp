//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_SnapshotCollectionObserver.hpp"

namespace MOR {

SnapshotCollectionObserver::SnapshotCollectionObserver(
    int period,
    const Teuchos::RCP<MultiVectorOutputFile> &snapshotFile) :
  snapshotCollector_(period, snapshotFile)
{
   // Nothing to do
}

void SnapshotCollectionObserver::observeSolution(const Epetra_Vector& solution)
{
  snapshotCollector_.addVector(0.0, solution);
}

void SnapshotCollectionObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  snapshotCollector_.addVector(time_or_param_val, solution);
}

} // namespace MOR
