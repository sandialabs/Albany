//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADD_FORCE_HPP
#define ADD_FORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace ATO {

template<typename EvalT, typename Traits>
class AddVector :  public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  AddVector(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> add_vector;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> w_bf;
  PHX::MDField<const ScalarT,Cell,Node,Dim> inResidual;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,Dim> outResidual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;

  bool projectFromQPs;
  bool negative;
  bool plusEquals;
};

template<typename EvalT, typename Traits>
class AddScalar :  public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  AddScalar(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT,Cell,QuadPoint> add_scalar;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> w_bf;
  PHX::MDField<const ScalarT,Cell,Node> inResidual;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> outResidual;

  std::size_t numNodes;
  std::size_t numQPs;

  bool projectFromQPs;
  bool negative;
  bool plusEquals;
};
}

#endif
