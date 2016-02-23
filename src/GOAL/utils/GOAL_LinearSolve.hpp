//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_LINEARSOLVE_H
#define GOAL_LINEARSOLVE_H

#include "Albany_DataTypes.hpp"

namespace Albany {
class Application;
}

namespace GOAL {

void solveLinearSystem(
    Teuchos::RCP<Albany::Application> app,
    Teuchos::RCP<Tpetra_CrsMatrix> A,
    Teuchos::RCP<Tpetra_Vector> x,
    Teuchos::RCP<Tpetra_Vector> b);

}

#endif
