//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MultiVectorInputFileFactory.hpp"

#include "MOR_MatrixMarketMVInputFile.hpp"
#include "MOR_Hdf5MVInputFile.hpp"
#include "MOR_ContainerUtils.hpp"

#include "Teuchos_Array.hpp"

#include "EpetraExt_ConfigDefs.h"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace MOR {

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
    result = rcp(new MatrixMarketMVInputFile(inputFileName_));
  }
  if (inputFileFormat_ == "HDF5") {
    const std::string groupName = params_->get("Input File Group Name", "default");
    result = rcp(new Hdf5MVInputFile(inputFileName_, groupName));
  }

  TEUCHOS_ASSERT(nonnull(result));
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
  const std::string defaultInputBaseFileName = params_->get("Input File Default Base File Name", "default_in");

  std::string defaultInputFilePostfix;
  if (inputFileFormat_ == "Matrix Market") defaultInputFilePostfix = "mtx";
  if (inputFileFormat_ == "HDF5")          defaultInputFilePostfix = "hdf5";
  const std::string defaultInputFileName = defaultInputBaseFileName + "." + defaultInputFilePostfix;

  const std::string inputFileName = !userInputFileName.empty() ? userInputFileName : defaultInputFileName;

  inputFileName_ = inputFileName;
}

} // namespace MOR
