//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_MeanSubstractingSnapshotPreprocessor.hpp"

#include "Teuchos_Assert.hpp"

namespace MOR {

MeanSubstractingSnapshotPreprocessor::MeanSubstractingSnapshotPreprocessor() :
  modifiedSnapshots_(),
  origin_()
{}

Teuchos::RCP<const Epetra_MultiVector>
MeanSubstractingSnapshotPreprocessor::modifiedSnapshotSet() const
{
  return modifiedSnapshots_;
}

Teuchos::RCP<const Epetra_Vector>
MeanSubstractingSnapshotPreprocessor::origin() const
{
  return origin_;
}

void
MeanSubstractingSnapshotPreprocessor::rawSnapshotSetIs(const Teuchos::RCP<Epetra_MultiVector> &rs)
{
  Teuchos::RCP<Epetra_Vector> snapshotMean;

  if (Teuchos::nonnull(rs)) {
    const int vecCount = rs->NumVectors();

    snapshotMean = Teuchos::rcp(new Epetra_Vector(rs->Map(), /* zeroOut =*/ true));
    for (int iVec = 0; iVec < vecCount; ++iVec) {
      snapshotMean->Update(1.0, *(*rs)(iVec), 1.0);
    }
    if (vecCount > 0) {
      snapshotMean->Scale(1.0 / vecCount);
    }

    for (int iVec = 0; iVec < vecCount; ++iVec) {
      (*rs)(iVec)->Update(-1.0, *snapshotMean, 1.0);
    }
  }

  origin_ = snapshotMean;
  modifiedSnapshots_ = rs;
}

} // namespace MOR
