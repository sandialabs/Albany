//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_EpetraMVSource.hpp"

#include "MOR_EpetraUtils.hpp"

namespace MOR {

Teuchos::RCP<Epetra_MultiVector>
BasicEpetraMVSource::truncatedMultiVectorNew(int vectorCountMax)
{
  // Inefficient default implementation:
  // Generate the full multivector then returns a truncated view
  const Teuchos::RCP<Epetra_MultiVector> fullMultiVector = this->multiVectorNew();
  return nonConstTruncatedView(fullMultiVector, vectorCountMax);
}

} // end namespace MOR
