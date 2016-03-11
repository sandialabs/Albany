//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_IdentityBasisSource.hpp"

#include "Epetra_MultiVector.h"

#include "Teuchos_Ptr.hpp"

namespace MOR {

IdentityBasisSource::IdentityBasisSource(const Epetra_Map &basisMap) :
  basisMap_(basisMap)
{}

ReducedBasisElements
IdentityBasisSource::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) {
  const Teuchos::RCP<Epetra_MultiVector> basis =
    Teuchos::rcp(new Epetra_MultiVector(basisMap_, basisMap_.NumGlobalElements(), /* zeroOut =*/ true));

  for (int i = 0; i < basis->NumVectors(); ++i) {
    const int lid = basisMap_.LID(i);
    if (lid != -1) {
      basis->ReplaceMyValue(lid, i, 1.0);
    }
  }

  return ReducedBasisElements(basis);
}

} // namespace MOR
