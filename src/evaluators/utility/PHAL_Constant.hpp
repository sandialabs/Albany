//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_CONSTANT_HPP
#define PHAL_CONSTANT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
class Constant : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {
  
public:
  
  Constant(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData ud);
  
private:
  
  typedef typename EvalT::ScalarT ScalarT;

  ScalarT value;

  PHX::MDField<ScalarT,Cell,Point> constant;

}; 
}

#endif
