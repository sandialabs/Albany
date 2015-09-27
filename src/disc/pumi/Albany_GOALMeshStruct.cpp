//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GOALMeshStruct.hpp"

Albany::GOALMeshStruct::GOALMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
		const Teuchos::RCP<const Teuchos_Comm>& commT) :
  PUMIMeshStruct(params, commT)
{
}

Albany::GOALMeshStruct::~GOALMeshStruct()
{
}

Albany::AbstractMeshStruct::msType
Albany::GOALMeshStruct::meshSpecsType()
{
  return GOAL_MS;
}

apf::Field*
Albany::GOALMeshStruct::createNodalField(char const* name, int valueType)
{
  return apf::createFieldOn(this->mesh, name, valueType);
}
