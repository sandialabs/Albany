//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_BCUTILS_HPP
#define GOAL_BCUTILS_HPP

#include "Albany_DataTypes.hpp"

namespace Albany {
class Application;
}

namespace GOAL {

void computeHierarchicBCs(
    Albany::Application const& app,
    Teuchos::RCP<Tpetra_Vector> const& res,
    Teuchos::RCP<Tpetra_CrsMatrix> const& jac);

}

#endif
