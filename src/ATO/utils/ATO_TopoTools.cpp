//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_TopoTools.hpp"
#include "Teuchos_TestForException.hpp"

namespace ATO {

//**********************************************************************
//TopoToolsFactory::TopoToolsFactory(const Teuchos::RCP<Teuchos::ParameterList>& topoParams)
//: _topoParams(topoParams) { }


//**********************************************************************
Teuchos::RCP<TopoTools> TopoToolsFactory::create(const Teuchos::ParameterList& topoParams)
{
  std::string pType = topoParams.get<std::string>("Penalization");
  if( pType == "SIMP" )  return Teuchos::rcp(new TopoTools_SIMP(topoParams));
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Penalization type " << pType << " Unknown!" << std::endl );
}

//**********************************************************************
TopoTools_SIMP::TopoTools_SIMP(const Teuchos::ParameterList& topoParams) 
{
  const Teuchos::ParameterList& simpParams = topoParams.get<Teuchos::ParameterList>("SIMP");
  penaltyParam = simpParams.get<double>("Penalization Parameter");
}

double TopoTools_SIMP::Penalize(double rho) { return pow(rho,penaltyParam);}
double TopoTools_SIMP::dPenalize(double rho) { return penaltyParam*pow(rho,penaltyParam-1.0);}


}

