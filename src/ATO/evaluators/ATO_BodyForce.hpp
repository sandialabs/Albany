//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_BODY_FORCE_HPP
#define ATO_BODY_FORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace ATO {

template<typename EvalT, typename Traits>
class BodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
                  public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  BodyForce(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  unsigned int numQPs;
  unsigned int numDims;

  Teuchos::Array<RealType> bodyForceVector;

  // Input:
  PHX::MDField<const ScalarT,Cell,QuadPoint> density;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> outVector;

};
}

#endif
