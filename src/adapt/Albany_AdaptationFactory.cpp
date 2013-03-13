//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Albany_AdaptationFactory.hpp"
#include "Albany_CopyRemesh.hpp"
#ifdef ALBANY_LCM
#include "Albany_TopologyModification.hpp"
#endif
#ifdef ALBANY_SCOREC
#include "Albany_MeshAdapt.hpp"
#endif

Albany::AdaptationFactory::AdaptationFactory(
       const Teuchos::RCP<Teuchos::ParameterList>& adaptParams_,
       const Teuchos::RCP<ParamLib>& paramLib_,
       Albany::StateManager& StateMgr_,
       const Teuchos::RCP<const Epetra_Comm>& comm_) :
  adaptParams(adaptParams_),
  paramLib(paramLib_),
  StateMgr(StateMgr_),
  comm(comm_)
{
}

Teuchos::RCP<Albany::AbstractAdapter>
Albany::AdaptationFactory::create()
{
  Teuchos::RCP<Albany::AbstractAdapter> strategy;
  using Teuchos::rcp;
  std::string& method = adaptParams->get("Method", "");

  if (method == "Copy Remesh") {
    strategy = rcp(new Albany::CopyRemesh(adaptParams, paramLib, StateMgr, comm));
  }
#ifdef ALBANY_LCM
  else if (method == "Topmod") {
    strategy = rcp(new Albany::TopologyMod(adaptParams, paramLib, StateMgr, comm));
  }
#endif
#ifdef ALBANY_SCOREC
  else if (method == "RPI Mesh Adapt") {
    strategy = rcp(new Albany::MeshAdapt<Albany::UnifSizeField>(adaptParams, paramLib, StateMgr, comm));
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       std::endl << 
                       "Error!  Unknown adaptivity method requested: " << method << 
                       "!" << std::endl << "Supplied parameter list is " << 
                       std::endl << *adaptParams);
  }

  return strategy;
}

