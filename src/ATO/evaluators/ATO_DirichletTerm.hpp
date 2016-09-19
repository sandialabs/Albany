//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_DIRICHLET_FORCE_HPP
#define ATO_DIRICHLET_FORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace ATO {

template<typename EvalT, typename Traits>
class DirichletVectorTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DirichletVectorTerm(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  unsigned int numQPs;
  unsigned int numDims;

  Teuchos::Array<RealType> constraintVector;
  Teuchos::Array<RealType> penaltyVector;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> outVector;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> dirVector;

};

template<typename EvalT, typename Traits>
class DirichletScalarTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DirichletScalarTerm(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  unsigned int numQPs;

  RealType constraintValue;
  RealType penaltyValue;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> outValue;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint> dirValue;

};
}

#endif
