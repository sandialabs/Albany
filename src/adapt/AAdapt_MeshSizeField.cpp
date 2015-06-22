//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

MeshSizeField::MeshSizeField(
    const Teuchos::RCP<Albany::APFDiscretization>& disc): 
    mesh_struct(disc->getAPFMeshStruct()),
    commT(disc->getComm())
{
}

}
