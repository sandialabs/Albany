//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_MultiVectorInputFile.hpp"

namespace MOR {

using ::Teuchos::RCP;
using ::Teuchos::rcp;

RCP<Epetra_MultiVector> MultiVectorInputFile::readPartial(const Epetra_Map &map, int maxVecCount) {
  // Inefficient default implementation
  // Reads the whole basis first and then copy a portion of it into the returned value
  const RCP<Epetra_MultiVector> fullBasis = read(map);
  if (fullBasis->NumVectors() <= maxVecCount) {
    return fullBasis;
  }
  return rcp(new Epetra_MultiVector(Copy, *fullBasis, 0, maxVecCount));
}

} // namespace MOR
