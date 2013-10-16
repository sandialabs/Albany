//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MatrixMarketMVInputFile.hpp"

#include "EpetraExt_MultiVectorIn.h"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"

#include <stdexcept>
#include <cstddef>

namespace MOR {

using Teuchos::RCP;
using Teuchos::rcp;
using EpetraExt::MatrixMarketFileToMultiVector;

MatrixMarketMVInputFile::MatrixMarketMVInputFile(const std::string &path) :
  MultiVectorInputFile(path)
{
  // Nothing to do
}

RCP<Epetra_MultiVector> MatrixMarketMVInputFile::read(const Epetra_Map &map)
{
  // Create an uninitialized raw pointer,
  // to be passed by reference // to MatrixMarketFileToMultiVector for initialization
  Epetra_MultiVector *raw_result = NULL;
  const int err = MatrixMarketFileToMultiVector(path().c_str(), map, raw_result);
  TEUCHOS_TEST_FOR_EXCEPTION(err != 0,
                             std::runtime_error,
                             "Cannot open input file: " + path());

  // Take ownership of the returned newly allocated object
  RCP<Epetra_MultiVector> result(raw_result);

  TEUCHOS_TEST_FOR_EXCEPT(result.is_null());
  return result;
}

} // namespace MOR
