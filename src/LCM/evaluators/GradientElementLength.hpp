//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GRADIENT_ELEMENT_LENGTH_HPP
#define GRADIENT_ELEMENT_LENGTH_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    Compute element length in the direction of the solution gradient
    (cf. Tezduyar and Park CMAME 1986).


*/

template<typename EvalT, typename Traits>
class GradientElementLength : public PHX::EvaluatorWithBaseImpl<Traits>,
	       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  GradientElementLength(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> unitScalarGradient;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;


  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> elementLength;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
};
}

#endif
