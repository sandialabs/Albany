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


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
PHAL::ResponseBaseCommon<EvalT, Traits>::
ResponseBaseCommon(Teuchos::ParameterList& p)
{
  std::string responseID = p.get<string>("Response ID");
  Teuchos::RCP<PHX::DataLayout> dummy_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout");
  
  response_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(responseID, dummy_dl));
  this->addEvaluatedField(*response_operation);

  responseIndex = p.get<int>("Response Index");
  responseFn    = p.get<Teuchos::RCP<Albany::EvaluatedResponseFunction> >("Response Function");
  
  this->setName(responseID + PHX::TypeString<EvalT>::value);
}



// **********************************************************************
//   RESIDUAL
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Residual,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Residual, Traits>::responseIndex;
  for(unsigned int i=0; i<local_g.size(); ++i)
    local_g[i] = (*(workset.responses))[respIndx][i];
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Residual, Traits>::responseIndex;
  for(unsigned int i=0; i<local_g.size(); ++i)
    (*(workset.responses))[respIndx][i] = local_g[i];
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{
  local_g.resize(initialVals.size());

  PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
    responseFn->setResponseInitialValues(initialVals);
}

// **********************************************************************
