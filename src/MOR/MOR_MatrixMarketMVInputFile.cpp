//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MatrixMarketMVInputFile.hpp"

#include "Epetra_Comm.h"

#include "EpetraExt_MultiVectorIn.h"
#include "EpetraExt_mmio.h"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"

#include <stdexcept>
#include <cstddef>
#include <cstdio>

namespace MOR {

MatrixMarketMVInputFile::MatrixMarketMVInputFile(const std::string &path) :
  MultiVectorInputFile(path)
{
  // Nothing to do
}

int MatrixMarketMVInputFile::readVectorCount(const Epetra_Comm &comm)
{
  const int masterPID = 0;

  int result; // Variable left unitialized
  {
    std::FILE * handle;
    {
      int ierr;
      if (comm.MyPID() == masterPID) {
        // Only master reads file
        handle = std::fopen(this->path().c_str(), "r");
        ierr = (handle == NULL);
      }

      const int info = comm.Broadcast(&ierr, 1, masterPID);
      TEUCHOS_TEST_FOR_EXCEPT(info != 0);

      TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0,
                                 std::runtime_error,
                                 "Cannot open input file: " + path());
    }

    {
      int ierr;
      if (comm.MyPID() == masterPID) {
        int dummy;
        ierr = EpetraExt::mm_read_mtx_array_size(handle, &dummy, &result);
      }

      const int info = comm.Broadcast(&ierr, 1, masterPID);
      TEUCHOS_TEST_FOR_EXCEPT(info != 0);

      TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0,
                                 std::runtime_error,
                                 "Error reading input file: " + path());
    }
  }

  // Master broadcasts result to slaves
  const int info = comm.Broadcast(&result, 1, masterPID);
  TEUCHOS_TEST_FOR_EXCEPT(info != 0);

  return result;
}

Teuchos::RCP<Epetra_MultiVector> MatrixMarketMVInputFile::read(const Epetra_Map &map)
{
  // Create a raw pointer to be passed by reference
  // to MatrixMarketFileToMultiVector for initialization
  Epetra_MultiVector *raw_result = NULL;
  const int ierr = EpetraExt::MatrixMarketFileToMultiVector(this->path().c_str(), map, raw_result);
  TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0,
                             std::runtime_error,
                             "Error reading input file: " + path());

  // Take ownership of the returned newly allocated object
  Teuchos::RCP<Epetra_MultiVector> result(raw_result);

  TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(result));
  return result;
}

} // namespace MOR
