//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MultiVectorInputFile.hpp"

#include "MOR_EpetraUtils.hpp"

namespace MOR {

using ::Teuchos::RCP;
using ::Teuchos::rcp;

RCP<Epetra_MultiVector> MultiVectorInputFile::readPartial(const Epetra_Map &map, int maxVecCount) {
  // Inefficient default implementation:
  // Read the whole basis first then returns a partial view of the truncated vectors
  const RCP<Epetra_MultiVector> fullBasis = this->read(map);
  return nonConstTruncatedView(fullBasis, maxVecCount);
}

} // namespace MOR
