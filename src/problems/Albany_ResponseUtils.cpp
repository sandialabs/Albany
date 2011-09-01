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


#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_ResponseUtils.hpp"

#include "PHAL_FactoryTraits.hpp"
#ifdef ALBANY_LCM       
#include "LCM_FactoryTraits.hpp"
#endif

Albany::ResponseUtils::ResponseUtils(
     Teuchos::RCP<Albany::Layouts> dl_, std::string facTraits_) :
     dl(dl_), facTraits(facTraits_)
{
   TEST_FOR_EXCEPTION(facTraits!="PHAL" && facTraits!="LCM", std::logic_error,
       "ResponseUtils constructor: unrecognized  facTraits flag: "<< facTraits);
}

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > 
Albany::ResponseUtils::constructResponses(
  std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::ParameterList& responseList, 
  std::map<string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build, 
  Albany::StateManager& stateMgr)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;
  using PHAL::AlbanyTraits;


   // Parameters for Response Evaluators
   //  Iterate through list of responses (from input xml file).  For each, create a response
   //  function and possibly a parameter list to construct a response evaluator.
   int num_responses = responseList.get("Number", 0);
   responses.resize(num_responses);

   std::map<string, RCP<ParameterList> > response_evaluators_to_build;
   std::vector<string> responseIDs_to_require;

   for (int i=0; i<num_responses; i++) 
   {
     std::string responseID = Albany::strint("Response",i);
     std::string name = responseList.get(responseID, "??");

     RCP<ParameterList> p;

     if( getStdResponseFn(name, i, responseList, responses, stateMgr, p) ) {
       if(p != Teuchos::null) {
	 response_evaluators_to_build[responseID] = p;
	 responseIDs_to_require.push_back(responseID);
       }
     }

     else {
       TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Error!  Unknown response function " << name <<
           "!" << std::endl << "Supplied parameter list is " <<
           std::endl << responseList);
     }
   } // end of loop over responses

   //! Create field manager for responses
   return createResponseFieldManager(response_evaluators_to_build, 
			      evaluators_to_build, responseIDs_to_require);
}

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > 
Albany::ResponseUtils::createResponseFieldManager(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& response_evaluators_to_build,
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build,
    const std::vector<std::string>& responseIDs_to_require)
{
  using Teuchos::RCP;
  using std::string;
  using PHAL::AlbanyTraits;

  // Build Response Evaluators for each evaluation type
  RCP< std::vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
    response_evaluators;
   
  response_evaluators_to_build.insert(evaluators_to_build.begin(), evaluators_to_build.end());

 
  if (facTraits=="PHAL") {
    PHX::EvaluatorFactory<AlbanyTraits,PHAL::FactoryTraits<AlbanyTraits> > factory;
    response_evaluators = factory.buildEvaluators(response_evaluators_to_build);
  }
#ifdef ALBANY_LCM       
  else if (facTraits=="LCM") {
    PHX::EvaluatorFactory<AlbanyTraits,LCM::FactoryTraits<AlbanyTraits> > factory;
    response_evaluators = factory.buildEvaluators(response_evaluators_to_build);
  }
#endif

  // Create a Response FieldManager
  Teuchos::RCP<PHX::FieldManager<AlbanyTraits> > rfm =
    Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

  // Register all Evaluators
  PHX::registerEvaluators(response_evaluators, *rfm);

  // Set required fields: ( Response<i>, dl->dummy ), for responses 
  //  evaluated by the response evaluators
  std::vector<string>::const_iterator it;
  for (it = responseIDs_to_require.begin(); it != responseIDs_to_require.end(); it++)
  {
    const string& responseID = *it;

    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::Residual>(res_response_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::Jacobian>(jac_response_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::Tangent>(tan_response_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::SGResidual>(sgres_response_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::SGJacobian>(sgjac_response_tag);
    PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::SGTangent>(sgtan_response_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::MPResidual>(mpres_response_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::MPJacobian>(mpjac_response_tag);
    PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_response_tag(responseID, dl->dummy);
    rfm->requireField<AlbanyTraits::MPTangent>(mptan_response_tag);
  }

  return rfm;
}


Teuchos::RCP<Teuchos::ParameterList>
Albany::ResponseUtils::setupResponseFnForEvaluator(
  Teuchos::ParameterList& responseList, int responseNumber,
  std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;

  std::string responseID = Albany::strint("Response",responseNumber);       
  std::string responseParamsID = Albany::strint("ResponseParams",responseNumber);       
  TEST_FOR_EXCEPTION(!responseList.isSublist(responseParamsID), 
		     Teuchos::Exceptions::InvalidParameter,
		     std::endl << Albany::strint("Response",responseNumber) <<
		     " requires a parameter list" << std::endl);

  ParameterList& responseParams = responseList.sublist(responseParamsID);
  RCP<ParameterList> p = rcp(new ParameterList);

  RCP<Albany::EvaluatedResponseFunction> 
    evResponse = Teuchos::rcp(new Albany::EvaluatedResponseFunction());
  responses[responseNumber] = evResponse;

  // Common parameters to all response evaluators
  p->set<string>("Response ID", responseID);
  p->set<int>   ("Response Index", responseNumber);
  p->set< RCP<Albany::EvaluatedResponseFunction> >("Response Function", evResponse);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);

  return p;
}


// - Returns true if responseName was recognized and response function constructed.
// - If p is non-Teuchos::null upon exit, then an evaluator should be build using
//   p as the parameter list. 
bool
Albany::ResponseUtils::getStdResponseFn(
    std::string responseName, int responseIndex,
    Teuchos::ParameterList& responseList,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
    Albany::StateManager& stateMgr,    
    Teuchos::RCP<Teuchos::ParameterList>& p)
{
  using std::string;
  using Teuchos::RCP;
  using PHX::DataLayout;
  using PHAL::AlbanyTraits;

  p = Teuchos::null;

  if (responseName == "Field Integral") 
  {
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<AlbanyTraits>::id_qcad_response_fieldintegral;
#ifdef ALBANY_LCM       
//    else if (facTraits=="LCM") type =  LCM::FactoryTraits<AlbanyTraits>::id_qcad_response_fieldintegral;
#endif

double length_unit_in_m=1.0e-6;  cout << "KACK lengthUniot " << endl;
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<int>("Type", type);
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set<double>("Length unit in m", length_unit_in_m);
    return true;
  }

  else if (responseName == "Field Value") 
  { 
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<AlbanyTraits>::id_qcad_response_fieldvalue;
#ifdef ALBANY_LCM       
//    else if (facTraits=="LCM") type =  LCM::FactoryTraits<AlbanyTraits>::id_qcad_response_fieldvalue;
#endif

    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<int>("Type", type);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    return true;
  }

  else if (responseName == "Center Of Mass") 
  { 
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<AlbanyTraits>::id_qcad_response_centerofmass;
#ifdef ALBANY_LCM       
//    else if (facTraits=="LCM") type =  LCM::FactoryTraits<AlbanyTraits>::id_qcad_response_fieldvalue;
#endif

    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<int>("Type", type);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    return true;
  }


  else if (responseName == "Save Field")
  { 
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<AlbanyTraits>::id_qcad_response_savefield;
#ifdef ALBANY_LCM       
//    else if (facTraits=="LCM") type =  LCM::FactoryTraits<AlbanyTraits>::id_qcad_response_savefield;
#endif
       
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<int>("Type", type);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Cell Scalar Data Layout", dl->cell_scalar);
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set<string>("Weights Name",   "Weights");
    return true;
  }

  else if (responseName == "Solution Average") {
    responses[responseIndex] = Teuchos::rcp(new Albany::SolutionAverageResponseFunction());
    return true;
  }

  else if (responseName == "Solution Two Norm") {
    responses[responseIndex] = Teuchos::rcp(new Albany::SolutionTwoNormResponseFunction());
    return true;
  }

  else return false; // responseName not recognized
}

