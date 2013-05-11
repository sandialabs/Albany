//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>

#include "Albany_AdaptationFactory.hpp"
#include "Albany_CopyRemesh.hpp"
#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)
#include "Albany_TopologyModification.hpp"
#include "Albany_RandomFracture.hpp"
#endif
#ifdef ALBANY_SCOREC
#include "Albany_MeshAdapt.hpp"
#endif

namespace Albany {

  //----------------------------------------------------------------------------
  Albany::AdaptationFactory::
  AdaptationFactory(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params,
                    const Teuchos::RCP<ParamLib>& param_lib,
                    Albany::StateManager& state_mgr,
                    const Teuchos::RCP<const Epetra_Comm>& comm) :
    adapt_params_(adapt_params),
    param_lib_(param_lib),
    state_mgr_(state_mgr),
    epetra_comm_(comm)
  {
  }
  //----------------------------------------------------------------------------
  Teuchos::RCP<Albany::AbstractAdapter>
  Albany::AdaptationFactory::createAdapter()
  {
    using Teuchos::rcp;

    Teuchos::RCP<Albany::AbstractAdapter> strategy;
    std::string& method = adapt_params_->get("Method", "");

    if (method == "Copy Remesh") {
      strategy = rcp(new Albany::CopyRemesh(adapt_params_,
                                            param_lib_,
                                            state_mgr_,
                                            epetra_comm_));
    }
#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)
    else if (method == "Topmod") {
      strategy = rcp(new Albany::TopologyMod(adapt_params_,
                                             param_lib_,
                                             state_mgr_,
                                             epetra_comm_));
    }
    else if (method == "Random") {
      strategy = rcp(new Albany::RandomFracture(adapt_params_,
                                                param_lib_,
                                                state_mgr_,
                                                epetra_comm_));
    }
#endif
#ifdef ALBANY_SCOREC
  else if (method == "RPI Unif Size") {
    strategy = rcp(new Albany::MeshAdapt<Albany::UnifSizeField>(adapt_params_, param_lib_, state_mgr_, epetra_comm_));
  }
  else if (method == "RPI UnifRef Size") {
    strategy = rcp(new Albany::MeshAdapt<Albany::UnifRefSizeField>(adapt_params_, param_lib_, state_mgr_, epetra_comm_));
  }
#endif
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,
                                 Teuchos::Exceptions::InvalidParameter,
                                 std::endl <<
                                 "Error! Unknown adaptivity method requested:"
                                 << method <<
                                 " !" << std::endl
                                 << "Supplied parameter list is " <<
                                 std::endl << *adapt_params_);
    }

    return strategy;
  }
  //----------------------------------------------------------------------------
}
