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

#include "Epetra_Comm.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_HDF5.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"

#include <stdexcept>
#include <string>

// Helper function
#include <algorithm>

template <typename Container, typename T>
bool contains(const Container &c, const T &t) {
  return std::find(c.begin(), c.end(), t) != c.end(); 
}

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;
using Teuchos::Array;

SnapshotCollection::SnapshotCollection(const RCP<ParameterList> &params) :
  params_(params),
  outputFileFormat_(),
  outputFileName_(),
  period_(),
  skipCount_(0)
{
  initOutputFileFormat();
  initOutputFileName();
  initPeriod();
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

    const std::string groupName = "snapshots";
    
    if (outputFileFormat_ == "Matrix Market")
    {
      EpetraExt::MultiVectorToMatrixMarketFile(outputFileName_.c_str(), collection, groupName.c_str());
    }

#ifdef HAVE_EPETRAEXT_HDF5
    if (outputFileFormat_ == "HDF5")
    {
      const Epetra_Comm &fileComm = collection.Comm();
      EpetraExt::HDF5 hdf5Output(fileComm);
      
      hdf5Output.Create(outputFileName_); // Truncate existing file if necessary
      hdf5Output.Write(groupName, collection);
      hdf5Output.Close();
    }
#endif /* HAVE_EPETRAEXT_HDF5 */
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

void SnapshotCollection::initOutputFileFormat()
{
  Array<std::string> validFileFormats;
  validFileFormats.append("Matrix Market");
#ifdef HAVE_EPETRAEXT_HDF5
  validFileFormats.append("HDF5");
#endif /* HAVE_EPETRAEXT_HDF5 */

  const std::string outputFileFormat = params_->get("Output File Format", validFileFormats[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(!contains(validFileFormats, outputFileFormat),
                             std::out_of_range,
                             outputFileFormat + " not in " + validFileFormats.toString());
  outputFileFormat_ = outputFileFormat;
}

void SnapshotCollection::initOutputFileName()
{
  const std::string userOutputFileName = params_->get("Output File Name", "");
  
  const std::string defaultOutputFilePrefix = "snap";
  std::string defaultOutputFilePostfix;
  if (outputFileFormat_ == "Matrix Market") defaultOutputFilePostfix = "mtx";
  if (outputFileFormat_ == "HDF5")          defaultOutputFilePostfix = "hdf5";
  const std::string defaultOutputFileName = defaultOutputFilePrefix + "." + defaultOutputFilePostfix; 

  const std::string outputFileName = !userOutputFileName.empty() ? userOutputFileName : defaultOutputFileName;
  
  outputFileName_ = outputFileName;
}

void SnapshotCollection::initPeriod()
{
  const std::size_t period = params_->get("Period", 1);
  TEUCHOS_TEST_FOR_EXCEPTION(period == 0,
                             std::out_of_range,
                             "period > 0");
  period_ = period;
}

} // end namespace Albany
