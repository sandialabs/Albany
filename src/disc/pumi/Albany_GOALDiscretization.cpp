//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMIDiscretization.hpp"
#include "Albany_GOALDiscretization.hpp"

Albany::GOALDiscretization::GOALDiscretization(
    Teuchos::RCP<Albany::GOALMeshStruct> meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_):
  PUMIDiscretization(meshStruct_, commT_, rigidBodyModes_)
{
  goalMeshStruct = meshStruct_;
}

Albany::GOALDiscretization::~GOALDiscretization()
{
}

int Albany::GOALDiscretization::getNumNodesPerElem(int ebi)
{
  return goalMeshStruct->getNumNodesPerElem(ebi);
}
