//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SimDiscretization.hpp"
#include <apfSIM.h>

Albany::SimDiscretization::SimDiscretization(
    Teuchos::RCP<Albany::SimMeshStruct> meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_):
  APFDiscretization(meshStruct_, commT_, rigidBodyModes_)
{
}

Albany::SimDiscretization::~SimDiscretization()
{
}

void Albany::SimDiscretization::
createField(const char* name, int value_type)
{
  apf::Mesh* mesh = meshStruct->getMesh();
  apf::Field* f = apf::createSIMField(mesh,name,value_type,mesh->getShape());
  apf::zeroField(f);
}
