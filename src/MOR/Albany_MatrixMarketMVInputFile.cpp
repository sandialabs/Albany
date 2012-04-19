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

#include "Albany_MatrixMarketMVInputFile.hpp"

#include "EpetraExt_MultiVectorIn.h"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"

#include <stdexcept>
#include <cstddef>

namespace Albany {

using Teuchos::RCP;
using Teuchos::rcp;
using EpetraExt::MatrixMarketFileToMultiVector;

MatrixMarketMVInputFile::MatrixMarketMVInputFile(const std::string &path) :
  MultiVectorInputFile(path)
{
  // Nothing to do
}

RCP<Epetra_MultiVector> MatrixMarketMVInputFile::vectorNew(const Epetra_Map &map)
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

} // end namespace Albany
