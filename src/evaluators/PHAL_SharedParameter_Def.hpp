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
#include "Sacado_ParameterRegistration.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
SharedParameter<EvalT, Traits>::
SharedParameter(const Teuchos::ParameterList& p) 
{  
  paramName =  p.get<std::string>("Parameter Name");
  paramValue =  p.get<double>("Parameter Value");

  Teuchos::RCP<PHX::DataLayout> layout =
      p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  //! Initialize field with same name as parameter
  PHX::MDField<ScalarT,Dim> f(paramName, layout);
  paramAsField = f;

  // Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); //, Teuchos::null ANDY - why a compiler error with this?
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      paramName, this, paramLib);

  this->addEvaluatedField(paramAsField);
  this->setName("Shared Parameter"+PHAL::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void SharedParameter<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(paramAsField,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SharedParameter<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  paramAsField(0) = paramValue;
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename SharedParameter<EvalT,Traits>::ScalarT& 
SharedParameter<EvalT,Traits>::getValue(const std::string &n)
{
  TEST_FOR_EXCEPT(n != paramName);
  return paramValue;
}

// **********************************************************************
}


