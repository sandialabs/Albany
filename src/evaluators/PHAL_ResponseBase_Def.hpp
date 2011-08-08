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
    local_g[i] = (*(workset.responses[respIndx]))[i];
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Residual, Traits>::responseIndex;
  for(unsigned int i=0; i<local_g.size(); ++i)
    (*(workset.responses[respIndx]))[i] = local_g[i];
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



// **********************************************************************
//   JACOBIAN
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Jacobian,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Jacobian, Traits>::responseIndex;
  for(unsigned int i=0; i<local_g.size(); ++i)
    local_g[i] = (*(workset.responses[respIndx]))[i];
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  int lcol, col;

  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Jacobian, Traits>::responseIndex;
  std::size_t numNodes = workset.responseDerivatives[respIndx]->GlobalLength(); //ANDY - check this vs. MyLength() ?
  for(unsigned int i=0; i<local_g.size(); ++i)
    (*(workset.responses[respIndx]))[i] = local_g[i].val();

  //add derivative information in local_g to workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < numNodes; ++node) {
      col  = nodeID[node][0]; // neq assumed == 1, otherwise would be firstCol, then offset by neq index
      lcol = node; //local column

      for(unsigned int i=0; i<local_g.size(); ++i) 
	workset.responseDerivatives[respIndx]->SumIntoMyValue(i, col, local_g[i].fastAccessDx(lcol));
    }
  }
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{
  local_g.resize(initialVals.size());

  PHAL::ResponseBase<PHAL::AlbanyTraits::Jacobian, Traits>::
    responseFn->setResponseInitialValues(initialVals);
}

// **********************************************************************


// **********************************************************************
//   TANGENT
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::Tangent, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Tangent,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Tangent, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Tangent, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Tangent, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::Tangent, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::Tangent, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{

}

// **********************************************************************


// **********************************************************************
//   STOCHASTIC GALERKIN RESIDUAL
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::SGResidual, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::SGResidual,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::SGResidual, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::SGResidual, Traits>::responseIndex;
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::SGResidual, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::SGResidual, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::SGResidual, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{

}

// **********************************************************************


// **********************************************************************
//   STOCHASTIC GALERKIN JACOBIAN
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::SGJacobian, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::SGJacobian,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::SGJacobian, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::SGJacobian, Traits>::responseIndex;
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::SGJacobian, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::SGJacobian, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::SGJacobian, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{

}

// **********************************************************************


// **********************************************************************
//   MULTI-POINT RESIDUAL
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::MPResidual, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::MPResidual,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::MPResidual, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::MPResidual, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::MPResidual, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::MPResidual, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::MPResidual, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{

}

// **********************************************************************


// **********************************************************************
//   MULTI-POINT JACOBIAN
// **********************************************************************

template<typename Traits>
PHAL::ResponseBase<PHAL::AlbanyTraits::MPJacobian, Traits>::
ResponseBase(Teuchos::ParameterList& p)
  : PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::MPJacobian,Traits>(p)
{
}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::MPJacobian, Traits>::
beginEvaluateFields(typename Traits::EvalData workset)
{
  //transfer workset to local_g
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::MPJacobian, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::MPJacobian, Traits>::
endEvaluateFields(typename Traits::EvalData workset)
{
  //transfer local_g to workset
  unsigned int respIndx = PHAL::ResponseBaseCommon<PHAL::AlbanyTraits::MPJacobian, Traits>::responseIndex;

}

// **********************************************************************

template<typename Traits>
void PHAL::ResponseBase<PHAL::AlbanyTraits::MPJacobian, Traits>::
setInitialValues(const std::vector<double>& initialVals)
{

}

// **********************************************************************


