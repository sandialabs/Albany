//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "LandIce_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"

#include "LandIce_ProblemUtils.hpp"
#include "LandIce_ResponseSurfaceVelocityMismatch.hpp"
#include "LandIce_ResponseSMBMismatch.hpp"
#include "LandIce_ResponseGLFlux.hpp"
#include "LandIce_ResponseBoundarySquaredL2Norm.hpp"

namespace LandIce {

template<typename EvalT, typename Traits>
ResponseUtilities<EvalT,Traits>::
ResponseUtilities(Teuchos::RCP<Albany::Layouts> dl_) :
  base_type(dl_)
{
}

template<typename EvalT, typename Traits>
Teuchos::RCP<const PHX::FieldTag>
ResponseUtilities<EvalT,Traits>::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  Teuchos::ParameterList& responseParams,
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem)
{
  auto ev_tag = base_type::constructResponses(fm,responseParams,paramsFromProblem);
  if (!ev_tag.is_null()) {
    return ev_tag;
  }

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using FST = FieldScalarType;

  std::string responseName = responseParams.get<std::string>("Name");
  RCP<ParameterList> p = rcp(new ParameterList);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set<RCP<ParameterList> >("Parameters From Problem", paramsFromProblem);
  RCP<PHX::Evaluator<Traits>> res_ev;

  if (responseName == "Surface Velocity Mismatch") {
    res_ev = rcp(new ResponseSurfaceVelocityMismatch<EvalT,Traits>(*p,this->dl));
  } else if (responseName == "Surface Mass Balance Mismatch") {
    auto thickness_st = paramsFromProblem->get<FST>("Ice Thickness Scalar Type",FST::Real);
    res_ev = createEvaluatorWithOneScalarType<ResponseSMBMismatch,EvalT>(p,this->dl,thickness_st);
  } else if (responseName == "Grounding Line Flux") {
    auto thickness_st = paramsFromProblem->get<FST>("Ice Thickness Scalar Type",FST::Real);
    res_ev = createEvaluatorWithOneScalarType<ResponseGLFlux,EvalT>(p,this->dl,thickness_st);
  } else if (responseName == "Boundary Squared L2 Norm") {
    res_ev = rcp(new ResponseBoundarySquaredL2Norm<EvalT,Traits>(*p,this->dl));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        "Error!  Unknown response function " + responseName + "!\n"
        "         Supplied parameter list is:\n" << responseParams);
  }

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

  return ev_tag;
}

} // namespace LandIce
