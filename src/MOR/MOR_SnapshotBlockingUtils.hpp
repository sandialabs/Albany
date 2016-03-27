//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOR_SNAPSHOTBLOCKINGUTILS_HPP
#define MOR_SNAPSHOTBLOCKINGUTILS_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayView.hpp"

namespace MOR {

Teuchos::RCP<Epetra_Vector> isolateUniformBlock(
    const Teuchos::ArrayView<const int> &myBlockLIDs,
    Epetra_MultiVector &snapshots);

} // end namespace MOR

#endif /*MOR_SNAPSHOTBLOCKINGUTILS_HPP*/
