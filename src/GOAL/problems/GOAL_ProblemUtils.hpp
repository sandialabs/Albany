//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_PROBLEMUTILS_HPP
#define GOAL_PROBLEMUTILS_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"

namespace Albany {
class MeshSpecsStruct;
}

namespace GOAL {

void enrichMeshSpecs(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > ms);

void decreaseMeshSpecs(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > ms);

}

#endif
