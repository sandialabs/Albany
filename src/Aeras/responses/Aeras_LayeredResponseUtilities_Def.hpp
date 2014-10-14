/*
 * Aeras_ResponseUtilitiesDef.hpp
 *
 *  Created on: Jul 30, 2014
 *      Author: swbova
 */

#include "Aeras/responses/Aeras_LayeredResponseUtilities.hpp"
#include "Albany_Utils.hpp"
#include "Aeras/problems/Aeras_Layouts.hpp"

#include "PHAL_SaveNodalField.hpp"
#include "Phalanx_FieldManager.hpp"
#include "Aeras_ShallowWaterResponseL2Error.hpp"
#include "Aeras/responses/Aeras_TotalVolume.hpp"
#include "Teuchos_RCP.hpp"

template<typename EvalT, typename Traits>
Aeras::LayeredResponseUtilities<EvalT,Traits>::LayeredResponseUtilities(
  Teuchos::RCP<Aeras::Layouts> dl_) :
  dl(dl_)
{
}

template<typename EvalT, typename Traits>
Teuchos::RCP<const PHX::FieldTag>
Aeras::LayeredResponseUtilities<EvalT,Traits>::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  Teuchos::ParameterList& responseParams,
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem,
  Albany::StateManager& stateMgr)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;

  std::string responseName = responseParams.get<std::string>("Name");
  RCP<ParameterList> p = rcp(new ParameterList);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set<RCP<ParameterList> >("Parameters From Problem", paramsFromProblem);

  Teuchos::RCP<const PHX::FieldTag> response_tag;

  if (responseName == "Aeras Total Volume")
  {
    RCP<Aeras::TotalVolume<EvalT,Traits> > res_ev =
      rcp(new Aeras::TotalVolume<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown response function " << responseName <<
      "!" << std::endl << "Supplied parameter list is " <<
      std::endl << responseParams);

  return response_tag;

}

