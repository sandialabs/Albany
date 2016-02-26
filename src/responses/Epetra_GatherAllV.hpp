//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

class Epetra_Comm;

namespace Epetra {

int GatherAllV(
    const Epetra_Comm &comm,
    const int *myVals, int myCount,
    int *allVals, int allCount);

} // namespace Epetra
