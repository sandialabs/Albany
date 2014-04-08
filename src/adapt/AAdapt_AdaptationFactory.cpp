//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>

#include "AAdapt_AdaptationFactory.hpp"
#include "AAdapt_CopyRemesh.hpp"
#if defined(ALBANY_LCM)
#include "AAdapt_TopologyModification.hpp"
#endif
#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)
#include "AAdapt_RandomFracture.hpp"
#endif
#if defined(ALBANY_LCM) && defined(ALBANY_STK_PERCEPT)
#include "AAdapt_STKAdapt.hpp"
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
                  const Teuchos::RCP<const Epetra_Comm>& comm) :
  adapt_params_(adapt_params),
  param_lib_(param_lib),
  state_mgr_(state_mgr),
  epetra_comm_(comm) {
}
//----------------------------------------------------------------------------
Teuchos::RCP<AAdapt::AbstractAdapter>
AAdapt::AdaptationFactory::createAdapter() {
  using Teuchos::rcp;

  Teuchos::RCP<AAdapt::AbstractAdapter> strategy;
  std::string& method = adapt_params_->get("Method", "");

  if(method == "Copy Remesh") {
    strategy = rcp(new AAdapt::CopyRemesh(adapt_params_,
                                          param_lib_,
                                          state_mgr_,
                                          epetra_comm_));
  }

#if defined(ALBANY_LCM)

  else if(method == "Topmod") {
    strategy = rcp(new AAdapt::TopologyMod(adapt_params_,
                                           param_lib_,
                                           state_mgr_,
                                           epetra_comm_));
  }

#endif

#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)

  else if(method == "Random") {
    strategy = rcp(new AAdapt::RandomFracture(adapt_params_,
                   param_lib_,
                   state_mgr_,
                   epetra_comm_));
  }

#endif
#ifdef ALBANY_SCOREC

  else if(method == "RPI Unif Size") {
    strategy = rcp(new AAdapt::MeshAdapt<AAdapt::UnifSizeField>(adapt_params_, param_lib_, state_mgr_, epetra_comm_));
  }

  else if(method == "RPI UnifRef Size") {
    strategy = rcp(new AAdapt::MeshAdapt<AAdapt::UnifRefSizeField>(adapt_params_, param_lib_, state_mgr_, epetra_comm_));
  }

#ifdef SCOREC_SPR
  else if(method == "RPI SPR Size") {
    strategy = rcp(new AAdapt::MeshAdapt<AAdapt::SPRSizeField>(adapt_params_, param_lib_, state_mgr_, epetra_comm_));
  }
#endif

#endif
#if defined(ALBANY_LCM) && defined(ALBANY_STK_PERCEPT)

  else if(method == "Unif Size") {
    strategy = rcp(new AAdapt::STKAdapt<AAdapt::STKUnifRefineField>(adapt_params_,
                   param_lib_,
                   state_mgr_,
                   epetra_comm_));
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
