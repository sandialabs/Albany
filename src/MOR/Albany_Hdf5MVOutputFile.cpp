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
