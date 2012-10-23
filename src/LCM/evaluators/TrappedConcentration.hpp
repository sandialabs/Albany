//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TRAPPED_CONCENTRATION_HPP
#define TRAPPED_CONCENTRATION_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    This evaluator computes the hydrogen concentration at trapped site
    through conservation of hydrogen atom

*/

template<typename EvalT, typename Traits>
class TrappedConcentration : public PHX::EvaluatorWithBaseImpl<Traits>,
	       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  TrappedConcentration(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint> Vmolar;
  PHX::MDField<ScalarT,Cell,QuadPoint> Clattice;
  PHX::MDField<ScalarT,Cell,QuadPoint> Ntrap;
  PHX::MDField<ScalarT,Cell,QuadPoint> Keq;

  ScalarT Nlattice;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> Ctrapped;

  unsigned int numQPs;
  unsigned int numDims;
};
}

#endif
