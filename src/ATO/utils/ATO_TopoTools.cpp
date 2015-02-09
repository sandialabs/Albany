//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "ATO_TopoTools.hpp"
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

  bounds = topoParams.get<Teuchos::Array<double> >("Bounds");

  const Teuchos::ParameterList& functionParams = topoParams.sublist("Functions");
  int nFunctions = functionParams.get<int>("Number of Functions");
  penaltyFunctions.resize(nFunctions);

  for(int i=0; i<nFunctions; i++){
    const Teuchos::ParameterList& fParams = functionParams.sublist(Albany::strint("Function",i));
    penaltyFunctions[i] = PenaltyFunction(fParams);
  }
}
 
//**********************************************************************
Topology::PenaltyFunction::PenaltyFunction(const Teuchos::ParameterList& fParams)
//**********************************************************************
{
  simp = Teuchos::null;
  ramp = Teuchos::null;

  std::string penalty = fParams.get<std::string>("Function Type");
  if( penalty == "SIMP" ){
    simp = Teuchos::rcp(new Simp(fParams));
    pType = SIMP;
  } else 
  if( penalty == "RAMP" ){
    ramp = Teuchos::rcp(new Ramp(fParams));
    pType = RAMP;
  } else
  if( penalty == "H1" ){
    h1 = Teuchos::rcp(new H1(fParams));
    pType = HONE;
  } else
  if( penalty == "H2" ){
    h2 = Teuchos::rcp(new H2(fParams));
    pType = HTWO;
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Function type " << penalty << " Unknown!" << std::endl );
}

//**********************************************************************
Simp::Simp(const Teuchos::ParameterList& fParams)
//**********************************************************************
{
  penaltyParam = fParams.get<double>("Penalization Parameter");
  if(fParams.isType<double>("Minimum")){
    minValue = fParams.get<double>("Minimum");
  } else minValue = 0.0;
}

//**********************************************************************
Ramp::Ramp(const Teuchos::ParameterList& fParams)
//**********************************************************************
{
  penaltyParam = fParams.get<double>("Penalization Parameter");
  if(fParams.isType<double>("Minimum")){
    minValue = fParams.get<double>("Minimum");
  } else minValue = 0.0;
}

//**********************************************************************
H1::H1(const Teuchos::ParameterList& fParams)
//**********************************************************************
{
  regLength = fParams.get<double>("Regularization Length");
  if(fParams.isType<double>("Minimum")){
    minValue = fParams.get<double>("Minimum");
  } else minValue = 0.0;
}

//**********************************************************************
H2::H2(const Teuchos::ParameterList& fParams)
//**********************************************************************
{
  regLength = fParams.get<double>("Regularization Length");
  if(fParams.isType<double>("Minimum")){
    minValue = fParams.get<double>("Minimum");
  } else minValue = 0.0;
}

} // end ATO namespace
