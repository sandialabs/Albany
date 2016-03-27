//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MultiVectorOutputFileFactory.hpp"

#include "MOR_MatrixMarketMVOutputFile.hpp"
#include "MOR_Hdf5MVOutputFile.hpp"
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

MultiVectorOutputFileFactory::MultiVectorOutputFileFactory(const Teuchos::RCP<Teuchos::ParameterList> &params) :
  params_(params)
{
  initValidFileFormats();
  initOutputFileFormat();
  initOutputFileName();
}

RCP<MultiVectorOutputFile> MultiVectorOutputFileFactory::create()
{
  RCP<MultiVectorOutputFile> result;
  if (outputFileFormat_ == "Matrix Market") {
    result = rcp(new MatrixMarketMVOutputFile(outputFileName_));
  }
  if (outputFileFormat_ == "HDF5") {
    const std::string groupName = params_->get("Output File Group Name", "default");
    result = rcp(new Hdf5MVOutputFile(outputFileName_, groupName));
  }

  TEUCHOS_ASSERT(nonnull(result));
  return result;
}

void MultiVectorOutputFileFactory::initValidFileFormats()
{
  validFileFormats_.append("Matrix Market");
#ifdef HAVE_EPETRAEXT_HDF5
  validFileFormats_.append("HDF5");
#endif /* HAVE_EPETRAEXT_HDF5 */
}

void MultiVectorOutputFileFactory::initOutputFileFormat()
{
  const std::string outputFileFormat = params_->get("Output File Format", validFileFormats_[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(!contains(validFileFormats_, outputFileFormat),
                             std::out_of_range,
                             outputFileFormat + " not in " + validFileFormats_.toString());
  outputFileFormat_ = outputFileFormat;
}

void MultiVectorOutputFileFactory::initOutputFileName()
{
  const std::string userOutputFileName = params_->get("Output File Name", "");
  const std::string defaultOutputBaseFileName = params_->get("Output File Default Base File Name", "default_out");

  std::string defaultOutputFilePostfix;
  if (outputFileFormat_ == "Matrix Market") defaultOutputFilePostfix = "mtx";
  if (outputFileFormat_ == "HDF5")          defaultOutputFilePostfix = "hdf5";
  const std::string defaultOutputFileName = defaultOutputBaseFileName + "." + defaultOutputFilePostfix;

  const std::string outputFileName = !userOutputFileName.empty() ? userOutputFileName : defaultOutputFileName;

  outputFileName_ = outputFileName;
}

} // namespace MOR
