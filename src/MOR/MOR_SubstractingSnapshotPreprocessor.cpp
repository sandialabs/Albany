//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_SubstractingSnapshotPreprocessor.hpp"

#include "Teuchos_Assert.hpp"

namespace MOR {

SubstractingSnapshotPreprocessor::SubstractingSnapshotPreprocessor(
    const Teuchos::RCP<const Epetra_Vector> &baseVector) :
  modifiedSnapshots_(),
  baseVector_(baseVector)
{}

Teuchos::RCP<const Epetra_MultiVector>
SubstractingSnapshotPreprocessor::modifiedSnapshotSet() const
{
  return modifiedSnapshots_;
}

Teuchos::RCP<const Epetra_Vector>
SubstractingSnapshotPreprocessor::baseVector() const
{
  return baseVector_;
}

void
SubstractingSnapshotPreprocessor::rawSnapshotSetIs(const Teuchos::RCP<Epetra_MultiVector> &rs)
{
  if (Teuchos::nonnull(rs) && Teuchos::nonnull(baseVector_)) {
    const int vecCount = rs->NumVectors();
    for (int iVec = 0; iVec < vecCount; ++iVec) {
      (*rs)(iVec)->Update(-1.0, *baseVector_, 1.0);
    }
  }
  modifiedSnapshots_ = rs;
}

} // namespace MOR
