//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LOADSTATEFIELD_HPP
#define PHAL_LOADSTATEFIELD_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief LoadStateField

*/

template<typename EvalT, typename Traits> 
class LoadStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  LoadStateField(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<ScalarT> data;
  std::string fieldName;
  std::string stateName;
};

}

#endif
