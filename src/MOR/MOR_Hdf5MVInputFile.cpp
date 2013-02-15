//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_Hdf5MVInputFile.hpp"

#include "Epetra_Comm.h"

#include "EpetraExt_HDF5.h"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"

#include <stdexcept>
#include <cstddef>

namespace MOR {

using Teuchos::RCP;
using Teuchos::rcp;

Hdf5MVInputFile::Hdf5MVInputFile(const std::string &path,
                                 const std::string &groupName) :
  MultiVectorInputFile(path),
  groupName_(groupName)
{
  // Nothing to do
}

RCP<Epetra_MultiVector> Hdf5MVInputFile::read(const Epetra_Map &map)
{
#ifdef HAVE_EPETRAEXT_HDF5
  const Epetra_Comm &fileComm = map.Comm();
  EpetraExt::HDF5 hdf5Input(fileComm);
  hdf5Input.Open(path(), H5F_ACC_RDONLY);
  
  TEUCHOS_TEST_FOR_EXCEPTION(!hdf5Input.IsOpen(),
                             std::runtime_error,
                             "Cannot open input file: " + path());
  TEUCHOS_TEST_FOR_EXCEPTION(!hdf5Input.IsContained(groupName_),
                             std::runtime_error,
                             "Cannot find source group name :" + groupName_ + " in file: " + path());

  // Create an uninitialized raw pointer,
  // to be passed by reference to HDF5::Read for initialization
  Epetra_MultiVector *raw_result = NULL;
  hdf5Input.Read(groupName_, map, raw_result);

  // Take ownership of the returned newly allocated object 
  RCP<Epetra_MultiVector> result = rcp(raw_result);
  
  hdf5Input.Close();

  TEUCHOS_TEST_FOR_EXCEPT(result.is_null());
  return result;
#else /* HAVE_EPETRAEXT_HDF5 */
  throw std::logic_error("HDF5 support disabled");
#endif /* HAVE_EPETRAEXT_HDF5 */
}

} // namespace MOR
