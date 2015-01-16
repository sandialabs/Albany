//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_TopoTools.hpp"
#include "ATO_TopoTools_Def.hpp"
#include "Teuchos_TestForException.hpp"

namespace ATO {

//**********************************************************************
Topology::Topology(const Teuchos::ParameterList& topoParams)
//**********************************************************************
{
  centering = topoParams.get<std::string>("Centering");
  name      = topoParams.get<std::string>("Topology Name");
  initValue = topoParams.get<double>("Initial Value");

  if( topoParams.isType<Teuchos::Array<std::string> >("Fixed Blocks") ){
    fixedBlocks = topoParams.get<Teuchos::Array<std::string> >("Fixed Blocks");
  }

  entityType = topoParams.get<std::string>("Entity Type");
  
  if( topoParams.isType<int>("Topology Output Filter") )
    topologyOutputFilter = topoParams.get<int>("Topology Output Filter");
  else topologyOutputFilter = -1;

  if( topoParams.isType<int>("Spatial Filter") )
    spatialFilterIndex = topoParams.get<int>("Spatial Filter");
  else spatialFilterIndex = -1;

 
  simp = Teuchos::null;
  ramp = Teuchos::null;

  std::string penalty = topoParams.get<std::string>("Penalization");
  if( penalty == "SIMP" ){
    simp = Teuchos::rcp(new Simp(topoParams));
    pType = SIMP;
    materialValue = simp->materialValue;
    voidValue = simp->voidValue;
  } else 
  if( penalty == "RAMP" ){
    ramp = Teuchos::rcp(new Ramp(topoParams));
    pType = RAMP;
    materialValue = ramp->materialValue;
    voidValue = ramp->voidValue;
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Penalization type " << penalty << " Unknown!" << std::endl );
}

//**********************************************************************
Simp::Simp(const Teuchos::ParameterList& topoParams)
//**********************************************************************
{
  const Teuchos::ParameterList& simpParams = topoParams.get<Teuchos::ParameterList>("SIMP");
  penaltyParam = simpParams.get<double>("Penalization Parameter");

  materialValue = 1.0;
  voidValue = 0.0;
}

//**********************************************************************
Ramp::Ramp(const Teuchos::ParameterList& topoParams)
//**********************************************************************
{
  const Teuchos::ParameterList& simpParams = topoParams.get<Teuchos::ParameterList>("RAMP");
  penaltyParam = simpParams.get<double>("Penalization Parameter");

  materialValue = 1.0;
  voidValue = 0.0;
}

} // end ATO namespace
