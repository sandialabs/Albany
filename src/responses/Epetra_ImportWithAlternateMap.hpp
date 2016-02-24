//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Epetra_CombineMode.h"

class Epetra_Import;
class Epetra_MultiVector;

namespace Epetra {

void
ImportWithAlternateMap(
    const Epetra_Import &importer,
    const Epetra_MultiVector &source,
    Epetra_MultiVector &target,
    Epetra_CombineMode mode);

} // namespace Epetra
