//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MatrixMarketMVOutputFile.hpp"

#include "EpetraExt_MultiVectorOut.h"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace MOR {

using Teuchos::RCP;
using EpetraExt::MultiVectorToMatrixMarketFile;

MatrixMarketMVOutputFile::MatrixMarketMVOutputFile(const std::string &path) :
  MultiVectorOutputFile(path)
{
  // Nothing to do
}

void MatrixMarketMVOutputFile::write(const Epetra_MultiVector &mv)
{
  // Write complete MultiVector (replace file if it exists)
  const int err = MultiVectorToMatrixMarketFile(path().c_str(), mv);

  TEUCHOS_TEST_FOR_EXCEPTION(err != 0,
                             std::runtime_error,
                             "Cannot create output file: " + path());
}

} // namespace MOR
