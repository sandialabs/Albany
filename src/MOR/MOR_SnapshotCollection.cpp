//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_SnapshotCollection.hpp"

#include "MOR_MultiVectorOutputFile.hpp"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace MOR {

SnapshotCollection::SnapshotCollection(
    int period,
    const Teuchos::RCP<MultiVectorOutputFile> &snapshotFile) :
  period_(period),
  snapshotFile_(snapshotFile),
  skipCount_(0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      period <= 0,
      std::out_of_range,
      "period = " << period << ", should have period > 0");
}

// TODO: Avoid doing real work in destructor
SnapshotCollection::~SnapshotCollection()
{
  const int vectorCount = snapshots_.size();
  if (vectorCount > 0)
  {
    const Epetra_Vector firstVector = snapshots_[0];
    const Epetra_BlockMap &map = firstVector.Map();
    Epetra_MultiVector collection(map, vectorCount);
    for (int iVec = 0; iVec < vectorCount; ++iVec)
    {
      *collection(iVec) = snapshots_[iVec];
    }

    snapshotFile_->write(collection);
  }
}

void SnapshotCollection::addVector(double stamp, const Epetra_Vector &value)
{
  if (skipCount_ == 0)
  {
    stamps_.push_back(stamp);
    snapshots_.push_back(value);
    skipCount_ = period_ - 1;
  }
  else
  {
    --skipCount_;
  }
}

} // namespace MOR
