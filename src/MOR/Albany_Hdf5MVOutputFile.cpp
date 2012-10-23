//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_Hdf5MVOutputFile.hpp"

#include "Epetra_Comm.h"

#include "EpetraExt_HDF5.h"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace Albany {

using Teuchos::RCP;
using Teuchos::rcp;

Hdf5MVOutputFile::Hdf5MVOutputFile(const std::string &path,
                                   const std::string &groupName) :
  MultiVectorOutputFile(path),
  groupName_(groupName)
{
  // Nothing to do
}

void Hdf5MVOutputFile::write(const Epetra_MultiVector &mv)
{
#ifdef HAVE_EPETRAEXT_HDF5
  const Epetra_Comm &fileComm = mv.Comm();
  EpetraExt::HDF5 hdf5Output(fileComm);
  
  hdf5Output.Create(path()); // Truncate existing file if necessary
  
  TEUCHOS_TEST_FOR_EXCEPTION(!hdf5Output.IsOpen(),
                             std::runtime_error,
                             "Cannot create output file: " + path());

  hdf5Output.Write(groupName_, mv);

  hdf5Output.Close();
#else /* HAVE_EPETRAEXT_HDF5 */
  throw std::logic_error("HDF5 support disabled");
#endif /* HAVE_EPETRAEXT_HDF5 */
}

} // end namespace Albany
