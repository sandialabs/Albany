//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_AbstractAdapter.hpp"

// Generic implementations that can be used by derived adapters

Albany::AbstractAdapter::AbstractAdapter(
         const Teuchos::RCP<Teuchos::ParameterList>& params_,
         const Teuchos::RCP<ParamLib>& paramLib_,
         Albany::StateManager& StateMgr_,
         const Teuchos::RCP<const Epetra_Comm>& comm_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  params(params_),
  paramLib(paramLib_),
  StateMgr(StateMgr_),
  comm(comm_)
{}

Teuchos::RCP<Teuchos::ParameterList>
Albany::AbstractAdapter::getGenericAdapterParams(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList(listname));

  validPL->set<std::string>("Method", "", "String to designate adapter class");

  return validPL;
}
