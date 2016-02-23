//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_SnapshotBlockingUtils.hpp"

#include "MOR_EpetraUtils.hpp"

namespace MOR {

Teuchos::RCP<Epetra_Vector> isolateUniformBlock(
    const Teuchos::ArrayView<const int> &myBlockLIDs,
    Epetra_MultiVector &snapshots)
{
  typedef Teuchos::ArrayView<const int>::const_iterator LIDIter;

  // Extract an orthogonal vector with non-zero entries corresponding to the block
  const Teuchos::RCP<Epetra_Vector> result(new Epetra_Vector(snapshots.Map(), /* zeroOut = */ true));
  for (LIDIter it = myBlockLIDs.begin(); it != myBlockLIDs.end(); ++it) {
    result->ReplaceMyValue(*it, 0, 1.0);
  }
  normalize(*result);

  // Zero-out the entries of the snapshots corresponding to the block
  const int vectorCount = snapshots.NumVectors();
  for (int iVec = 0; iVec < vectorCount; ++iVec) {
    for (LIDIter it = myBlockLIDs.begin(); it != myBlockLIDs.end(); ++it) {
      snapshots.ReplaceMyValue(*it, iVec, 0.0);
    }
  }

  return result;
}

} // end namespace MOR
