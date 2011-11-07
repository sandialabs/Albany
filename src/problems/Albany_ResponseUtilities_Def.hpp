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
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_SolutionFileL2ResponseFunction.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"

#include "QCAD_ResponseFieldIntegral.hpp"
#include "QCAD_ResponseFieldValue.hpp"
#include "QCAD_ResponseSaddleValue.hpp"
#include "QCAD_ResponseSaveField.hpp"
#include "QCAD_ResponseCenterOfMass.hpp"

template<typename EvalT, typename Traits>
Albany::ResponseUtilities<EvalT,Traits>::ResponseUtilities(
     Teuchos::RCP<Albany::Layouts> dl_) :
     dl(dl_)
{
}

template<typename EvalT, typename Traits>
void
Albany::ResponseUtilities<EvalT,Traits>::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::ParameterList& responseList, 
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

   std::vector<string> responseIDs_to_require;

   // First, add in response targets for PHAL_SaveStateField evaluators created
   //  during problem setup (these evaluators only act for residual type)
   if(typeid(EvalT) == typeid(PHAL::AlbanyTraits::Residual))
     responseIDs_to_require = stateMgr.getResidResponseIDsToRequire();

   // Now, loop over all Responses in the input file
   for (int i=0; i<num_responses; i++) 
   {
     std::string responseID = Albany::strint("Response",i);
     std::string name = responseList.get(responseID, "??");

     Teuchos::RCP< PHX::Evaluator<Traits> > ev;
     if( getStdResponseFn(name, i, responseList, responses, stateMgr, ev) ) {
       if(ev != Teuchos::null) {
         fm0.template registerEvaluator<EvalT>(ev);
	 responseIDs_to_require.push_back(responseID);
       }
     }

     else {
       TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Error!  Unknown response function " << name <<
           "!" << std::endl << "Supplied parameter list is " <<
           std::endl << responseList);
     }
   } // end of loop over responses

   //! Create field manager for responses
   createResponseFieldManager(fm0, responseIDs_to_require);
}

template<typename EvalT, typename Traits>
void
Albany::ResponseUtilities<EvalT,Traits>::createResponseFieldManager(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const std::vector<std::string>& responseIDs_to_require)
{
  using std::string;

  // Set required fields: ( Response<i>, dl->dummy ), for responses 
  //  evaluated by the response evaluators
  std::vector<string>::const_iterator it;
  for (it = responseIDs_to_require.begin(); it != responseIDs_to_require.end(); it++)
  {
    const string& responseID = *it;
cout << "RRRQ requiring: " << responseID << endl;
    PHX::Tag<typename EvalT::ScalarT> res_response_tag(responseID, dl->dummy);
    fm0.requireField<EvalT>(res_response_tag);
  }
}


template<typename EvalT, typename Traits>
Teuchos::RCP<Teuchos::ParameterList>
Albany::ResponseUtilities<EvalT,Traits>::setupResponseFnForEvaluator(
  Teuchos::ParameterList& responseList, int responseNumber,
  Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;

  std::string responseID = Albany::strint("Response",responseNumber);       
  std::string responseParamsID = Albany::strint("ResponseParams",responseNumber);       
  TEUCHOS_TEST_FOR_EXCEPTION(!responseList.isSublist(responseParamsID), 
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
template<typename EvalT, typename Traits>
bool
Albany::ResponseUtilities<EvalT,Traits>::getStdResponseFn(
    std::string responseName, int responseIndex,
    Teuchos::ParameterList& responseList,
    Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
    Albany::StateManager& stateMgr,    
    Teuchos::RCP< PHX::Evaluator<PHAL::AlbanyTraits> >& ev)
{
  using std::string;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using PHX::DataLayout;
  using PHAL::AlbanyTraits;

  RCP<Teuchos::ParameterList> p = Teuchos::null;

  if (responseName == "Field Integral") 
  {
    double length_unit_in_m=1.0e-6; 
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set<double>("Length unit in m", length_unit_in_m);
    ev = rcp(new QCAD::ResponseFieldIntegral<EvalT,Traits>(*p));

    return true;
  }

  else if (responseName == "Field Value") 
  { 
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    ev = rcp(new QCAD::ResponseFieldValue<EvalT,Traits>(*p));
    return true;
  }

  else if (responseName == "Center Of Mass") 
  { 
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    ev = rcp(new QCAD::ResponseCenterOfMass<EvalT,Traits>(*p));
    return true;
  }


  else if (responseName == "Save Field")
  { 
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Cell Scalar Data Layout", dl->cell_scalar);
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set<string>("Weights Name",   "Weights");
    ev = rcp(new QCAD::ResponseSaveField<EvalT,Traits>(*p));
    return true;
  }

  else if (responseName == "Solution Average") {
    responses[responseIndex] = rcp(new Albany::SolutionAverageResponseFunction());
    return true;
  }

  else if (responseName == "Solution Two Norm") {
    responses[responseIndex] = rcp(new Albany::SolutionTwoNormResponseFunction());
    return true;
  }

  else if (responseName == "Solution Max Value") {
    std::string responseParamsID = Albany::strint("ResponseParams",responseIndex);       
    Teuchos::ParameterList& responseParams = responseList.sublist(responseParamsID);

    // These are now redundantly specified -- need to fix when moved to evaluators for responses
    int eq = responseParams.get("Equation", 0);
    int neq = responseParams.get("Num Equations", 1);
    bool inor =  responseParams.get("Interleaved Ordering", true);
    
    responses[responseIndex] = rcp(new Albany::SolutionMaxValueResponseFunction(neq, eq, inor));
    return true;
  }

  else if (responseName == "Solution Two Norm File") {
    responses[responseIndex] = rcp(new Albany::SolutionFileL2ResponseFunction());
    return true;
  }

  else return false; // responseName not recognized
}

