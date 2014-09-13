//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SHAREDPARAMETER_HPP
#define PHAL_SHAREDPARAMETER_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"


namespace PHAL {
/** \brief SharedParameter

*/

template<typename EvalT, typename Traits> 
class SharedParameter : public PHX::EvaluatorWithBaseImpl<Traits>,
			public PHX::EvaluatorDerived<EvalT, Traits>,
			public Sacado::ParameterAccessor<EvalT, SPL_Traits>   {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  
  SharedParameter(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);
  
private:
  std::string paramName;
  ScalarT     paramValue;
  PHX::MDField<ScalarT,Dim> paramAsField;
};

}

#endif
