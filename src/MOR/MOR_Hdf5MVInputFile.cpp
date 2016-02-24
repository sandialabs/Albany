//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

namespace Detail {

#ifdef HAVE_EPETRAEXT_HDF5
void openReadOnlyAndCheckGroupName(
    EpetraExt::HDF5 &handle,
    const std::string &path,
    const std::string &groupName)
{
  handle.Open(path, H5F_ACC_RDONLY);

  TEUCHOS_TEST_FOR_EXCEPTION(!handle.IsOpen(),
      std::runtime_error,
      "Cannot open input file: " + path);

  TEUCHOS_TEST_FOR_EXCEPTION(!handle.IsContained(groupName),
      std::runtime_error,
      "Cannot find source group name :" + groupName + " in file: " + path);
}
#endif /* HAVE_EPETRAEXT_HDF5 */

} // namespace Detail


Hdf5MVInputFile::Hdf5MVInputFile(const std::string &path,
                                 const std::string &groupName) :
  MultiVectorInputFile(path),
  groupName_(groupName)
{
  // Nothing to do
}

int Hdf5MVInputFile::readVectorCount(const Epetra_Comm &comm)
{
#ifdef HAVE_EPETRAEXT_HDF5
  EpetraExt::HDF5 hdf5Input(comm);
  Detail::openReadOnlyAndCheckGroupName(hdf5Input, this->path(), groupName_);

  int result, dummy;
  hdf5Input.ReadMultiVectorProperties(groupName_, dummy, result);

  hdf5Input.Close();

  return result;
#else /* HAVE_EPETRAEXT_HDF5 */
  throw std::logic_error("HDF5 support disabled");
#endif /* HAVE_EPETRAEXT_HDF5 */
}

Teuchos::RCP<Epetra_MultiVector> Hdf5MVInputFile::read(const Epetra_Map &map)
{
#ifdef HAVE_EPETRAEXT_HDF5
  EpetraExt::HDF5 hdf5Input(map.Comm());
  Detail::openReadOnlyAndCheckGroupName(hdf5Input, this->path(), groupName_);

  // Create an uninitialized raw pointer,
  // to be passed by reference to HDF5::Read for initialization
  Epetra_MultiVector *raw_result = NULL;
  hdf5Input.Read(groupName_, map, raw_result);

  // Take ownership of the returned newly allocated object
  Teuchos::RCP<Epetra_MultiVector> result = Teuchos::rcp(raw_result);

  hdf5Input.Close();

  TEUCHOS_TEST_FOR_EXCEPT(result.is_null());
  return result;
#else /* HAVE_EPETRAEXT_HDF5 */
  throw std::logic_error("HDF5 support disabled");
#endif /* HAVE_EPETRAEXT_HDF5 */
}

} // namespace MOR
