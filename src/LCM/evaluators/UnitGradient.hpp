//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef UNIT_GRADIENT_HPP
#define UNIT_GRADIENT_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    Compute solution gradient unit vector.


*/

template<typename EvalT, typename Traits>
class UnitGradient : public PHX::EvaluatorWithBaseImpl<Traits>,
	       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  UnitGradient(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> scalarGrad;


  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> unitScalarGradient;

  unsigned int numQPs;
  unsigned int numDims;
};
}

#endif
