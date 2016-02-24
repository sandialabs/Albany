//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Epetra_ImportWithAlternateMap.hpp"

#include "Epetra_Import.h"
#include "Epetra_MultiVector.h"

#include "Teuchos_Assert.hpp"

namespace Epetra {

void
ImportWithAlternateMap(
    const Epetra_Import &importer,
    const Epetra_MultiVector &source,
    Epetra_MultiVector &target,
    Epetra_CombineMode mode)
{
  const Epetra_BlockMap savedMap = target.Map();
  {
    const int ierr = target.ReplaceMap(importer.TargetMap());
    TEUCHOS_ASSERT(ierr == 0);
  }
  {
    const int ierr = target.Import(source, importer, mode);
    TEUCHOS_ASSERT(ierr == 0);
  }
  {
    const int ierr = target.ReplaceMap(savedMap);
    TEUCHOS_ASSERT(ierr == 0);
  }
}

} // namespace Epetra
