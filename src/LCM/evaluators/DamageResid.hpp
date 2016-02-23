//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef DAMAGERESID_HPP
#define DAMAGERESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief Damage Equation Evaluator

    This evaluator computes the residual for the damage equation.

*/

template<typename EvalT, typename Traits>
class DamageResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DamageResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage_dot;
  PHX::MDField<ScalarT,Cell,QuadPoint> damageLS;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> damage_grad;
  PHX::MDField<ScalarT,Cell,QuadPoint> source;
  RealType gc;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> dResidual;

  bool enableTransient;
  unsigned int numQPs, numDims;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> flux;
};
}

#endif
