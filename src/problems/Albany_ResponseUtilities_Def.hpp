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

#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"

#include "QCAD_ResponseFieldIntegral.hpp"
#include "QCAD_ResponseFieldValue.hpp"
#include "QCAD_ResponseSaddleValue.hpp"
#include "QCAD_ResponseSaveField.hpp"
#include "QCAD_ResponseCenterOfMass.hpp"
#include "QCAD_ResponseSaddleValue.hpp"
#include "PHAL_ResponseFieldIntegral.hpp"


template<typename EvalT, typename Traits>
Albany::ResponseUtilities<EvalT,Traits>::ResponseUtilities(
  Teuchos::RCP<Albany::Layouts> dl_) :
  dl(dl_)
{
}

template<typename EvalT, typename Traits>
Teuchos::RCP<const PHX::FieldTag>
Albany::ResponseUtilities<EvalT,Traits>::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  Teuchos::ParameterList& responseParams, 
  Albany::StateManager& stateMgr)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;

  string responseName = responseParams.get<string>("Name");
  RCP<ParameterList> p = rcp(new ParameterList);
  p->set<ParameterList*>("Parameter List", &responseParams);
  Teuchos::RCP<const PHX::FieldTag> response_tag;

  if (responseName == "Field Integral") 
  {
    double length_unit_in_m=1.0e-6; 
    p->set<double>("Length unit in m", length_unit_in_m);
    RCP<QCAD::ResponseFieldIntegral<EvalT,Traits> > res_ev = 
      rcp(new QCAD::ResponseFieldIntegral<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Field Value") 
  { 
    RCP<QCAD::ResponseFieldValue<EvalT,Traits> > res_ev = 
      rcp(new QCAD::ResponseFieldValue<EvalT,Traits>(*p,dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Center Of Mass") 
  { 
    RCP<QCAD::ResponseCenterOfMass<EvalT,Traits> > res_ev = 
      rcp(new QCAD::ResponseCenterOfMass<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Save Field")
  { 
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<QCAD::ResponseSaveField<EvalT,Traits> > res_ev = 
      rcp(new QCAD::ResponseSaveField<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Saddle Value")
  {
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);  
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    RCP<QCAD::ResponseSaddleValue<EvalT,Traits> > res_ev = 
      rcp(new QCAD::ResponseSaddleValue<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "PHAL Field Integral") 
  {
    RCP<PHAL::ResponseFieldIntegral<EvalT,Traits> > res_ev = 
      rcp(new PHAL::ResponseFieldIntegral<EvalT,Traits>(*p, dl));
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

