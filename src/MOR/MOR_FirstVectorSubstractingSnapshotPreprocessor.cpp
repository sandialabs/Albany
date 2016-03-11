//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_FirstVectorSubstractingSnapshotPreprocessor.hpp"

#include "MOR_EpetraUtils.hpp"

#include "Teuchos_Assert.hpp"

namespace MOR {

FirstVectorSubstractingSnapshotPreprocessor::FirstVectorSubstractingSnapshotPreprocessor() :
  modifiedSnapshots_(),
  origin_()
{}

Teuchos::RCP<const Epetra_MultiVector>
FirstVectorSubstractingSnapshotPreprocessor::modifiedSnapshotSet() const
{
  return modifiedSnapshots_;
}

Teuchos::RCP<const Epetra_Vector>
FirstVectorSubstractingSnapshotPreprocessor::origin() const
{
  return origin_;
}

void
FirstVectorSubstractingSnapshotPreprocessor::rawSnapshotSetIs(const Teuchos::RCP<Epetra_MultiVector> &rs)
{
  const Teuchos::RCP<const Epetra_Vector> firstSnapshot = headView(rs);
  const Teuchos::RCP<const Epetra_MultiVector> snapshotRemainder = tailView(rs);

  if (Teuchos::nonnull(snapshotRemainder)) {
    const int vecCount = rs->NumVectors();
    for (int iVec = 1; iVec < vecCount; ++iVec) {
      (*rs)(iVec)->Update(-1.0, *firstSnapshot, 1.0);
    }
  }

  origin_ = firstSnapshot;
  modifiedSnapshots_ = snapshotRemainder;
}

} // namespace MOR
