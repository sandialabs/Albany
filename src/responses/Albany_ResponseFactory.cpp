//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_ResponseFactory.hpp"

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionValuesResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_SolutionMinValueResponseFunction.hpp"
#include "Albany_QuadraticLinearOperatorBasedResponseFunction.hpp"
#include "Albany_CumulativeScalarResponseFunction.hpp"
#include "Albany_ScalarResponsePower.hpp"
#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include "Albany_SolutionResponseFunction.hpp"
#include "Albany_WeightedMisfitResponseFunction.hpp"

#include "Albany_StringUtils.hpp"

#include "Teuchos_TestForException.hpp"

void
Albany::ResponseFactory::
createResponseFunction(
  const std::string& name,
  Teuchos::ParameterList& responseParams,
  Teuchos::Array< Teuchos::RCP<AbstractResponseFunction> >& responses) const
{
  using std::string;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using Teuchos::Array;

  RCP<const Teuchos_Comm> comm = app->getComm();

  if (name == "Solution Average") {
    responses.push_back(rcp(new Albany::SolutionAverageResponseFunction(comm)));
  }

  else if (name == "Solution Two Norm") {
    responses.push_back(rcp(new Albany::SolutionTwoNormResponseFunction(comm)));
  }
  else if (name == "Solution Values") {
    responses.push_back(rcp(new Albany::SolutionValuesResponseFunction(app, responseParams)));
  }
  else if (name == "Weighted Misfit") {
    responses.push_back(rcp(new Albany::WeightedMisfitResponse(app, responseParams)));
  }

  else if (name == "Solution Max Value") {
    int eq = responseParams.get("Equation", 0);
    int neq = app->getNumEquations();
    DiscType inor =  meshSpecs[0]->interleavedOrdering;

    responses.push_back(
      rcp(new Albany::SolutionMaxValueResponseFunction(comm, neq, eq, inor)));
  }

  else if (name == "Solution Min Value") {
    int eq = responseParams.get("Equation", 0);
    int neq = app->getNumEquations();
    DiscType inor =  meshSpecs[0]->interleavedOrdering;

    responses.push_back(
      rcp(new Albany::SolutionMinValueResponseFunction(comm, neq, eq, inor)));
  }

  else if (name == "Quadratic Linear Operator Based") {
    responses.push_back(
      rcp(new Albany::QuadraticLinearOperatorBasedResponseFunction(app, responseParams)));
  }

  else if (name == "Power Of Response") {
    double target = responseParams.get<double>("Target");
    double exponent = responseParams.get<double>("Exponent");
    Array< RCP<AbstractResponseFunction> > aggregated_responses;

    Teuchos::ParameterList sublist = responseParams.sublist("Response 0");
    std::string name = sublist.get<std::string>("Name");
    createResponseFunction(name, sublist, aggregated_responses);

    TEUCHOS_TEST_FOR_EXCEPTION(
        aggregated_responses[0]->isScalarResponse() != true, std::logic_error,
        "Response function 0 is not a scalar response function." <<
        std::endl <<
        "The power response can only uses scalar response " << "functions!");
    RCP<ScalarResponseFunction> scalar_response = Teuchos::rcp_dynamic_cast<ScalarResponseFunction>(aggregated_responses[0]);

    responses.push_back(rcp(new Albany::ScalarResponsePower(comm, scalar_response, target, exponent)));
  }

  else if (name == "Sum Of Responses") {
    int num_responses = responseParams.get<int>("Number Of Responses");
    Array< RCP<AbstractResponseFunction> > aggregated_responses;
    Array< RCP<ScalarResponseFunction> > scalar_responses;
    Array<double> scalar_weights;
    for (int i=0; i<num_responses; i++) {
      std::string id = util::strint("Response",i);
      Teuchos::ParameterList sublist = responseParams.sublist(id);
      std::string name = sublist.get<std::string>("Name");
      createResponseFunction(name, sublist, aggregated_responses);
    }
    scalar_responses.resize(aggregated_responses.size());
    scalar_weights.resize(aggregated_responses.size());
    for (int i=0; i<aggregated_responses.size(); i++) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          aggregated_responses[i]->isScalarResponse() != true, std::logic_error,
          "Response function " << i << " is not a scalar response function." <<
          std::endl <<
          "The aggregated response can only aggregate scalar response " << "functions!");
      scalar_responses[i] = Teuchos::rcp_dynamic_cast<ScalarResponseFunction>(aggregated_responses[i]);

      std::string id = util::strint("Scaling Coefficient",i);
      scalar_weights[i] = responseParams.get<double>(id, 1.0);
    }
    responses.push_back(rcp(new Albany::CumulativeScalarResponseFunction(comm, scalar_responses, scalar_weights)));
  }

  else if (name == "Field Integral" ||
     name == "Squared L2 Difference Source ST Target ST" ||
     name == "Squared L2 Difference Source ST Target MST" ||
     name == "Squared L2 Difference Source ST Target PST" ||
     name == "Squared L2 Difference Source PST Target ST" ||
     name == "Squared L2 Difference Source PST Target MST" ||
     name == "Squared L2 Difference Source PST Target PST" ||
     name == "Squared L2 Difference Source MST Target ST" ||
     name == "Squared L2 Difference Source MST Target MST" ||
     name == "Squared L2 Difference Source MST Target PST" ||
     name == "Squared L2 Difference Side Source ST Target ST" ||
     name == "Squared L2 Difference Side Source ST Target MST" ||
     name == "Squared L2 Difference Side Source ST Target PST" ||
     name == "Squared L2 Difference Side Source ST Target RT" ||
     name == "Squared L2 Difference Side Source PST Target ST" ||
     name == "Squared L2 Difference Side Source PST Target MST" ||
     name == "Squared L2 Difference Side Source PST Target PST" ||
     name == "Squared L2 Difference Side Source PST Target RT" ||
     name == "Squared L2 Difference Side Source MST Target ST" ||
     name == "Squared L2 Difference Side Source MST Target MST" ||
     name == "Squared L2 Difference Side Source MST Target PST" ||
     name == "Squared L2 Difference Side Source MST Target RT" ||
     name == "Surface Velocity Mismatch" ||
     name == "Surface Mass Balance Mismatch" ||
     name == "Grounding Line Flux" ||
     name == "Boundary Squared L2 Norm" ||
     name == "Region Boundary" ||
     name == "Stiffness Objective" ||
     name == "Interface Energy" ||
     name == "Internal Energy Objective" ||
     name == "Tensor PNorm Objective" ||
     name == "Tensor Average Response" ||
     name == "Homogenized Constants Response" ||
     name == "Modal Objective") {
    for (int i=0; i<meshSpecs.size(); i++) {
      responses.push_back(
          rcp(new Albany::FieldManagerScalarResponseFunction(
              app, prob, meshSpecs[i], stateMgr, responseParams)));
    }
  } else if (name == "Solution") {
    responses.push_back(
      rcp(new Albany::SolutionResponseFunction(app, responseParams)));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown response function " << name <<
      "!" << std::endl << "Supplied parameter list is " <<
      std::endl << responseParams);
  }
}

Teuchos::Array< Teuchos::RCP<Albany::AbstractResponseFunction> >
Albany::ResponseFactory::
createResponseFunctions(Teuchos::ParameterList& responseList) const
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using Teuchos::Array;

  int num_response_vecs = responseList.get<int>("Number Of Responses");
  Array<RCP<AbstractResponseFunction> > responses;

  for (int i=0; i<num_response_vecs; i++) {
    std::string sublist_name = util::strint("Response",i);
    ParameterList& response_params =
      responseList.sublist(sublist_name);
    std::string responseType = response_params.isParameter("Type") ?
        response_params.get<std::string>("Type") :
        std::string("Scalar Response");
    if (responseType == "Sum Of Responses")
      createResponseFunction(responseType, response_params, responses);
    else {
      std::string response_name = response_params.get<std::string>("Name");
      createResponseFunction(response_name, response_params, responses);
    }
  }

  return responses;
}
