//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"

#include "PHAL_ResponseSquaredL2Difference.hpp"
#include "PHAL_ResponseSquaredL2DifferenceSide.hpp"

namespace Albany {

template<typename EvalT, typename Traits>
ResponseUtilities<EvalT,Traits>::ResponseUtilities(
  Teuchos::RCP<Layouts> dl_) :
  dl(dl_)
{
}

template<typename EvalT, typename Traits>
Teuchos::RCP<const PHX::FieldTag>
ResponseUtilities<EvalT,Traits>::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  Teuchos::ParameterList& responseParams,
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem,
  StateManager& stateMgr)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;

  std::string responseName = responseParams.get<std::string>("Name");
  RCP<ParameterList> p = rcp(new ParameterList);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set<RCP<ParameterList> >("Parameters From Problem", paramsFromProblem);
  RCP<PHX::Evaluator<Traits>> res_ev;

  if (responseName == "Squared L2 Difference Source ST Target ST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSST_TST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source ST Target MST")
  {
    res_ev =rcp(new PHAL::ResponseSquaredL2DifferenceSST_TMST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source ST Target PST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSST_TPST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source PST Target ST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSPST_TST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source PST Target MST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSPST_TMST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source PST Target PST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSPST_TPST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source MST Target ST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSMST_TST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source MST Target MST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSMST_TMST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Source MST Target PST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSMST_TPST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source ST Target ST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source ST Target MST")
  {
    res_ev =rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TMST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source ST Target PST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TPST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source ST Target RT")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSST_TRT<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source PST Target ST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source PST Target MST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TMST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source PST Target PST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TPST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source PST Target RT")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSPST_TRT<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source MST Target ST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source MST Target MST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TMST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source MST Target PST")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TPST<EvalT,Traits>(*p,dl));
  }
  else if (responseName == "Squared L2 Difference Side Source MST Target RT")
  {
    res_ev = rcp(new PHAL::ResponseSquaredL2DifferenceSideSMST_TRT<EvalT,Traits>(*p,dl));
  }

  Teuchos::RCP<const PHX::FieldTag> ev_tag;
  if (!res_ev.is_null()) {

    // Register the evaluator
    fm.template registerEvaluator<EvalT>(res_ev);

    // Fetch the response tag. Usually it is the tag of the first evaluated field
    ev_tag = res_ev->evaluatedFields()[0];

    // The response tag is not the same of the evaluated field tag for PHAL::ScatterScalarResponse
    Teuchos::RCP<PHAL::ScatterScalarResponseBase<EvalT,Traits>> sc_resp;
    sc_resp = Teuchos::rcp_dynamic_cast<PHAL::ScatterScalarResponseBase<EvalT,Traits>>(res_ev);
    if (sc_resp!=Teuchos::null)
    {
      ev_tag = sc_resp->getResponseFieldTag();
    }

    // Require the response tag;
    fm.requireField<EvalT>(*ev_tag);
  }

  return ev_tag;
}

} // namespace Albany
