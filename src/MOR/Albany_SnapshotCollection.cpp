//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_SnapshotCollection.hpp"

#include "Albany_MultiVectorOutputFile.hpp"
#include "Albany_MultiVectorOutputFileFactory.hpp"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>
#include <string>

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;

SnapshotCollection::SnapshotCollection(const RCP<ParameterList> &params) :
  params_(fillDefaultParams(params)),
  snapshotFileFactory_(params),
  period_(),
  skipCount_(0)
{
  initPeriod();
}

RCP<ParameterList> SnapshotCollection::fillDefaultParams(const RCP<ParameterList> &params)
{
  params->get("Output File Group Name", "snapshots");
  params->get("Output File Default Base File Name", "snapshots");
  return params;
}

void SnapshotCollection::initPeriod()
{
  const std::size_t period = params_->get("Period", 1);
  TEUCHOS_TEST_FOR_EXCEPTION(period == 0,
                             std::out_of_range,
                             "period > 0");
  period_ = period;
}

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

    const RCP<MultiVectorOutputFile> outFile = snapshotFileFactory_.create(); 
    outFile->write(collection);
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

} // end namespace Albany
