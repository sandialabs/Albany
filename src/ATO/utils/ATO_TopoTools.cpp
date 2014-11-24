//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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

  
  if( topoParams.isType<int>("Topology Output Filter") )
    topologyOutputFilter = topoParams.get<int>("Topology Output Filter");
  else topologyOutputFilter = -1;

  if( topoParams.isType<int>("Spatial Filter") )
    spatialFilterIndex = topoParams.get<int>("Spatial Filter");
  else spatialFilterIndex = -1;
}


//**********************************************************************
Teuchos::RCP<Topology> TopologyFactory::create(const Teuchos::ParameterList& topoParams)
//**********************************************************************
{
  std::string pType = topoParams.get<std::string>("Penalization");
  if( pType == "SIMP" )  return Teuchos::rcp(new Topology_SIMP(topoParams));
  else
  if( pType == "RAMP" )  return Teuchos::rcp(new Topology_RAMP(topoParams));
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Penalization type " << pType << " Unknown!" << std::endl );
}

//**********************************************************************
Topology_SIMP::Topology_SIMP(const Teuchos::ParameterList& topoParams) : Topology(topoParams) 
//**********************************************************************
{
  const Teuchos::ParameterList& simpParams = topoParams.get<Teuchos::ParameterList>("SIMP");
  penaltyParam = simpParams.get<double>("Penalization Parameter");

  materialValue = 1.0;
  voidValue = 0.0;
}

double Topology_SIMP::Penalize(double rho) { return pow(rho,penaltyParam);}
double Topology_SIMP::dPenalize(double rho) { return penaltyParam*pow(rho,penaltyParam-1.0);}

//**********************************************************************
Topology_RAMP::Topology_RAMP(const Teuchos::ParameterList& topoParams) : Topology(topoParams) 
//**********************************************************************
{
  const Teuchos::ParameterList& simpParams = topoParams.get<Teuchos::ParameterList>("RAMP");
  penaltyParam = simpParams.get<double>("Penalization Parameter");

  materialValue = 1.0;
  voidValue = 0.0;
}

double Topology_RAMP::Penalize(double rho) { return rho/(1.0+penaltyParam*(1.0-rho)); }
double Topology_RAMP::dPenalize(double rho) { return (1.0+penaltyParam)/pow(1.0+penaltyParam*(1.0-rho),2.0); }

} // end ATO namespace
