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

#include "Albany_MatrixMarketMVOutputFile.hpp"

#include "EpetraExt_MultiVectorOut.h"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace Albany {

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

} // end namespace Albany
