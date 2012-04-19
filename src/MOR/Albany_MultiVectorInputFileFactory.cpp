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

#include "Albany_MultiVectorInputFileFactory.hpp"

#include "Albany_MatrixMarketMVInputFile.hpp"
#include "Albany_Hdf5MVInputFile.hpp"

#include "Teuchos_Array.hpp"

#include "EpetraExt_ConfigDefs.h"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

// Helper function
#include <algorithm>

template <typename Container, typename T>
bool contains(const Container &c, const T &t)
{
  return std::find(c.begin(), c.end(), t) != c.end(); 
}

namespace Albany {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;
using Teuchos::Array;

MultiVectorInputFileFactory::MultiVectorInputFileFactory(const Teuchos::RCP<Teuchos::ParameterList> &params) :
  params_(params)
{
  initValidFileFormats();
  initInputFileFormat();
  initInputFileName();
}

RCP<MultiVectorInputFile> MultiVectorInputFileFactory::create()
{
  RCP<MultiVectorInputFile> result;
  if (inputFileFormat_ == "Matrix Market") {
    result =  rcp(new MatrixMarketMVInputFile(inputFileName_));
  }
  if (inputFileFormat_ == "HDF5") {
    const std::string groupName = "basis"; // TODO not hardcoded
    result = rcp(new Hdf5MVInputFile(inputFileName_, groupName));
  }

  TEUCHOS_ASSERT(!result.is_null());
  return result;
}

void MultiVectorInputFileFactory::initValidFileFormats()
{
  validFileFormats_.append("Matrix Market");
#ifdef HAVE_EPETRAEXT_HDF5
  validFileFormats_.append("HDF5");
#endif /* HAVE_EPETRAEXT_HDF5 */
}

void MultiVectorInputFileFactory::initInputFileFormat()
{
  const std::string inputFileFormat = params_->get("Input File Format", validFileFormats_[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(!contains(validFileFormats_, inputFileFormat),
                             std::out_of_range,
                             inputFileFormat + " not in " + validFileFormats_.toString());
  inputFileFormat_ = inputFileFormat;
}

void MultiVectorInputFileFactory::initInputFileName() {
  const std::string userInputFileName = params_->get("Input File Name", "");
 
  const std::string defaultInputFilePrefix = "basis";
  std::string defaultInputFilePostfix;
  if (inputFileFormat_ == "Matrix Market") defaultInputFilePostfix = "mtx";
  if (inputFileFormat_ == "HDF5")          defaultInputFilePostfix = "hdf5";
  const std::string defaultInputFileName = defaultInputFilePrefix + "." + defaultInputFilePostfix;

  const std::string inputFileName = !userInputFileName.empty() ? userInputFileName : defaultInputFileName;

  inputFileName_ = inputFileName;
}

} // end namespace Albany
