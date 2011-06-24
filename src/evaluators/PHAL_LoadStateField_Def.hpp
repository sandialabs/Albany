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


#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
LoadStateField<EvalT, Traits>::
LoadStateField(const Teuchos::ParameterList& p) 
{  
  fieldName =  p.get<std::string>("Field Name");
  stateName =  p.get<std::string>("State Name");

  PHX::MDField<ScalarT> f(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("State Field Layout") );
  data = f;

  this->addEvaluatedField(data);
  this->setName("Load State Field"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void LoadStateField<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(data,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LoadStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  //cout << "LoadStateField importing state " << stateName << " to field " 
  //     << fieldName << " with size " << data.size() << endl;

  //ANDY: use oldState instead?? I don't think it matters - use const then?
  // Get state field container of same name
  Albany::StateVariables& newState = *workset.newState;
  Intrepid::FieldContainer<RealType>& stateToLoad  = *newState[stateName];

  for (int i=0; i < data.size() ; ++i) data[i] = stateToLoad[i]; 
}

// **********************************************************************
}

