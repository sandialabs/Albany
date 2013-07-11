//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_FirstVectorSubstractingSnapshotPreprocessor.hpp"

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
  Teuchos::RCP<const Epetra_Vector> firstSnapshot;
  Teuchos::RCP<const Epetra_MultiVector> snapshotRemainder;

  if (Teuchos::nonnull(rs)) {
    const int vecCount = rs->NumVectors();

    if (vecCount > 0) {
      firstSnapshot =
        Teuchos::rcpWithEmbeddedObjPostDestroy(new Epetra_Vector(View, *rs, 0), rs.getConst());

      const int remainderVecCount = vecCount - 1;

      if (remainderVecCount > 0) {
        for (int iVec = 1; iVec < vecCount; ++iVec) {
          (*rs)(iVec)->Update(-1.0, *firstSnapshot, 1.0);
        }

        snapshotRemainder =
          Teuchos::rcpWithEmbeddedObjPostDestroy(new Epetra_MultiVector(View, *rs, 1, remainderVecCount), rs.getConst());
      }
    }
  }

  origin_ = firstSnapshot;
  modifiedSnapshots_ = snapshotRemainder;
}

} // namespace MOR
