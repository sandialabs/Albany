//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef EQUILIBRIUMCONSTANT_HPP
#define EQUILIBRIUMCONSTANT_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    This evaluator computes equilibrium constant at intergration points.
    K_{T} = \exp(W_{B} / RT)

*/

template<typename EvalT, typename Traits>
class EquilibriumConstant : public PHX::EvaluatorWithBaseImpl<Traits>,
	       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  EquilibriumConstant(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint> Wbind;
  PHX::MDField<ScalarT,Cell,QuadPoint> temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint> Rideal;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> equilibriumConstant;

  unsigned int numQPs;
};
}

#endif
