//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_STK3DPointStruct.hpp"

//Constructor for meshes read from ASCII file
Albany::STK3DPointStruct::STK3DPointStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                                           const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct(params,Teuchos::null,3)
{
  // SetupFieldData(commT, neq_, req, sis, worksetSize);
  // metaData->commit();
  // bulkData->modification_begin(); // Begin modifying the mesh
  // buildMesh(commT);
  // bulkData->modification_end();

}
