//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_AbstractAdapter.hpp"

// Generic implementations that can be used by derived adapters

namespace Albany {

  //----------------------------------------------------------------------------
  Albany::AbstractAdapter::
  AbstractAdapter(const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<ParamLib>& param_lib,
                  Albany::StateManager& state_mgr,
                  const Teuchos::RCP<const Epetra_Comm>& comm) :
    output_stream_(Teuchos::VerboseObjectBase::getDefaultOStream()),
    adapt_params_(params),
    param_lib_(param_lib),
    state_mgr_(state_mgr),
    epetra_comm_(comm)
  {}

  //----------------------------------------------------------------------------
  Teuchos::RCP<Teuchos::ParameterList>
  Albany::AbstractAdapter::
  getGenericAdapterParams(std::string listname) const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      Teuchos::rcp(new Teuchos::ParameterList(listname));

    valid_pl->set<std::string>("Method",
                              "",
                              "String to designate adapter class");

    return valid_pl;
  }
  //----------------------------------------------------------------------------
}
