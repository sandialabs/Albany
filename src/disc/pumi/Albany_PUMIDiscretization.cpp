//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_APFDiscretization.hpp"
#include "Albany_PUMIDiscretization.hpp"

Albany::PUMIDiscretization::PUMIDiscretization(
    Teuchos::RCP<Albany::PUMIMeshStruct> meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_):
  APFDiscretization(meshStruct_, commT_, rigidBodyModes_)
{
  pumiMeshStruct = meshStruct_;
}

Albany::PUMIDiscretization::~PUMIDiscretization()
{
}

void Albany::PUMIDiscretization::
createField(const char* name, int value_type)
{
  apf::Field* f =
    apf::createFieldOn(meshStruct->getMesh(), name, value_type);
  apf::zeroField(f);
}
