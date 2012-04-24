/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

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
