//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
  this->registerSacadoParameter(paramName, paramLib);

  this->addEvaluatedField(paramAsField);
  this->setName("Shared Parameter" );
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
  TEUCHOS_TEST_FOR_EXCEPT(n != paramName);
  return paramValue;
}

// **********************************************************************

template<typename EvalT, typename Traits>
SharedParameterVec<EvalT, Traits>::
SharedParameterVec(const Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> layout = p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); //, Teuchos::null ANDY - why a compiler error with this?

  numParams = layout->dimension(1);

  paramNames.resize(numParams);
  paramValues.resize(numParams);
  paramNames = p.get<Teuchos::Array<std::string>>("Parameters Names");
  paramValues = p.get<Teuchos::Array<ScalarT>>("Parameters Values");

  TEUCHOS_TEST_FOR_EXCEPTION (paramNames.size()==numParams, std::logic_error,
                              "Error! The array of names' size does not match the layout first dimension.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (paramValues.size()==numParams, std::logic_error,
                              "Error! The array of values' size does not match the layout first dimension.\n");


  std::string paramVecName = p.get<std::string>("Parameter Vector Name");

  this->registerSacadoParameter(paramVecName, paramLib);

  this->addEvaluatedField(paramAsField);

  this->setName("Shared Parameter Vector" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SharedParameterVec<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(paramAsField,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SharedParameterVec<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i<numParams; ++i)
    paramAsField(i) = paramValues[i];
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename SharedParameterVec<EvalT,Traits>::ScalarT&
SharedParameterVec<EvalT,Traits>::getValue(const std::string &n)
{
  for (int i=0; i<numParams; ++i)
    if (n==paramNames[i])
      return paramValues[i];

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Parameter name not found.\n");

  // To avoid warnings
  static ScalarT dummy;
  return dummy;
}

// **********************************************************************
} // namespace PHAL
