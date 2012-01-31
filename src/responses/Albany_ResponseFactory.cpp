/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Albany_ResponseFactory.hpp"

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_SolutionFileResponseFunction.hpp"
#include "Albany_AggregateScalarResponseFunction.hpp"
#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include "Albany_SolutionResponseFunction.hpp"
#include "QCAD_SaddleValueResponseFunction.hpp"

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

  RCP<const Epetra_Comm> comm = app->getComm();

  if (name == "Solution Average") {
    responses.push_back(rcp(new Albany::SolutionAverageResponseFunction(comm)));
  }

  else if (name == "Solution Two Norm") {
    responses.push_back(rcp(new Albany::SolutionTwoNormResponseFunction(comm)));
  }

  else if (name == "Solution Max Value") {
    int eq = responseParams.get("Equation", 0);
    int neq = responseParams.get("Num Equations", 1);
    bool inor =  responseParams.get("Interleaved Ordering", true);
    
    responses.push_back(
      rcp(new Albany::SolutionMaxValueResponseFunction(comm, neq, eq, inor)));
  }

  else if (name == "Solution Two Norm File") {
    responses.push_back(
      rcp(new Albany::SolutionFileResponseFunction<Albany::NormTwo>(comm)));
  }

  else if (name == "Solution Inf Norm File") {
    responses.push_back(
      rcp(new Albany::SolutionFileResponseFunction<Albany::NormInf>(comm)));
  }

  else if (name == "Aggregated") {
    int num_aggregate_responses = responseParams.get<int>("Number");
    Array< RCP<AbstractResponseFunction> > aggregated_responses;
    Array< RCP<ScalarResponseFunction> > scalar_responses;
    for (int i=0; i<num_aggregate_responses; i++) {
      std::string id = Albany::strint("Response",i);
      std::string name = responseParams.get<std::string>(id);
      std::string sublist_name = Albany::strint("ResponseParams",i);
      createResponseFunction(name, responseParams.sublist(sublist_name),
			     aggregated_responses);

    }
    scalar_responses.resize(aggregated_responses.size());
    for (int i=0; i<aggregated_responses.size(); i++)
      scalar_responses[i] = 
	Teuchos::rcp_dynamic_cast<ScalarResponseFunction>(
	  aggregated_responses[i]);
    responses.push_back(
      rcp(new Albany::AggregateScalarResponseFunction(comm, scalar_responses)));
  }

  else if (name == "Field Integral" ||
	   name == "Field Value" ||
	   name == "Center Of Mass" ||
	   name == "Save Field" ||
	   name == "PHAL Field Integral") {
    responseParams.set("Name", name);
    for (int i=0; i<meshSpecs.size(); i++) {
      responses.push_back(
	rcp(new Albany::FieldManagerScalarResponseFunction(
	      app, prob, meshSpecs[i], stateMgr, responseParams)));
    }
  }

  else if (name == "Solution") {
    responses.push_back(
      rcp(new Albany::SolutionResponseFunction(app, responseParams)));
  }

  else if (name == "Saddle Value") {
    responseParams.set("Name", name);
    for (int i=0; i<meshSpecs.size(); i++) {
      responses.push_back(
	rcp(new QCAD::SaddleValueResponseFunction(
	      app, prob, meshSpecs[i], stateMgr, responseParams)));
    }
  }

  else {
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

  // First check for the old response specification
  if (responseList.isType<int>("Number")) {
    int num_aggregate_responses = responseList.get<int>("Number");
    if (num_aggregate_responses > 0) {
      Array<RCP<AbstractResponseFunction> > responses;
      createResponseFunction("Aggregated", responseList, responses);
      return responses;
    }
  }

  int num_response_vecs = responseList.get("Number of Response Vectors", 0);
  Array<RCP<AbstractResponseFunction> > responses;

  for (int i=0; i<num_response_vecs; i++) {
    std::string sublist_name = Albany::strint("Response Vector",i);
    ParameterList& response_params = 
      responseList.sublist(sublist_name);
    std::string response_name = response_params.get<std::string>("Name");
    createResponseFunction(response_name, response_params, responses);
  }

  return responses;
}
