//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//class Teuchos_Comm;

#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"


namespace Tpetra {

int GatherAllV(
    const Teuchos::RCP<const Teuchos::Comm<int> >& commT,
    const int *myVals, int myCount,
    int *allVals, int allCount);

} // namespace Tpetra
