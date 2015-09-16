//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <Teuchos_TestForException.hpp>

#include "AAdapt_AdaptationFactory.hpp"
#if defined(HAVE_STK)
#include "AAdapt_CopyRemesh.hpp"
#if defined(ALBANY_LCM) && defined(ALBANY_BGL)
#include "AAdapt_TopologyModification.hpp"
#endif
#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)
#include "AAdapt_RandomFracture.hpp"
#endif
#if defined(ALBANY_LCM) && defined(ALBANY_STK_PERCEPT)
#include "AAdapt_STKAdapt.hpp"
#endif
#endif
#ifdef ALBANY_SCOREC
#include "AAdapt_MeshAdapt.hpp"
#endif

namespace AAdapt {

//----------------------------------------------------------------------------
AAdapt::AdaptationFactory::
AdaptationFactory(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params,
                  const Teuchos::RCP<ParamLib>& param_lib,
                  Albany::StateManager& state_mgr,
                  const Teuchos::RCP<const Teuchos_Comm>& commT) :
  adapt_params_(adapt_params),
  param_lib_(param_lib),
  state_mgr_(state_mgr),
  commT_(commT) {
}
//----------------------------------------------------------------------------
Teuchos::RCP<AAdapt::AbstractAdapter>
AAdapt::AdaptationFactory::createAdapter() {
  using Teuchos::rcp;

  Teuchos::RCP<AAdapt::AbstractAdapter> strategy;
  std::string& method = adapt_params_->get("Method", "");

#if defined(HAVE_STK)
  if(method == "Copy Remesh") {
    strategy = rcp(new AAdapt::CopyRemesh(adapt_params_,
                                          param_lib_,
                                          state_mgr_,
                                          commT_));
  }

#if defined(ALBANY_LCM) && defined(ALBANY_BGL)

  else if(method == "Topmod") {
    strategy = rcp(new AAdapt::TopologyMod(adapt_params_,
                                           param_lib_,
                                           state_mgr_,
                                           commT_));
  }

#endif

#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)

  else if(method == "Random") {
    strategy = rcp(new AAdapt::RandomFracture(adapt_params_,
                   param_lib_,
                   state_mgr_,
                   commT_));
  }

#endif
#endif
#if 0
#if defined(ALBANY_LCM) && defined(ALBANY_STK_PERCEPT)

  else if(method == "Unif Size") {
    strategy = rcp(new AAdapt::STKAdapt<AAdapt::STKUnifRefineField>(adapt_params_,
                   param_lib_,
                   state_mgr_,
                   epetra_comm_));
  }

#endif
#endif

#if defined(HAVE_STK)
  else 
#endif
    TEUCHOS_TEST_FOR_EXCEPTION(true,
                               Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error! Unknown adaptivity method requested:"
                               << method <<
                               " !" << std::endl
                               << "Supplied parameter list is " <<
                               std::endl << *adapt_params_);

  return strategy;
}
//----------------------------------------------------------------------------
}
