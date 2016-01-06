//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef J2STRESS_HPP
#define J2STRESS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief J2Stress stress response

    This evaluator computes stress based on a uncoupled J2Stress
    potential

*/

template<typename EvalT, typename Traits>
class J2Stress : public PHX::EvaluatorWithBaseImpl<Traits>,
		 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  J2Stress(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  typename EvalT::ScalarT norm(Intrepid2::FieldContainer<ScalarT>);
  void exponential_map(Intrepid2::FieldContainer<ScalarT> &, const Intrepid2::FieldContainer<ScalarT>);

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;
  PHX::MDField<ScalarT,Cell,QuadPoint> yieldStrength;
  PHX::MDField<ScalarT,Cell,QuadPoint> hardeningModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> satMod;
  PHX::MDField<ScalarT,Cell,QuadPoint> satExp;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Fp;
  PHX::MDField<ScalarT,Cell,QuadPoint> eqps;

  std::string fpName, eqpsName;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // scratch space FCs
  Intrepid2::FieldContainer<ScalarT> be;
  Intrepid2::FieldContainer<ScalarT> s;
  Intrepid2::FieldContainer<ScalarT> N;
  Intrepid2::FieldContainer<ScalarT> A;
  Intrepid2::FieldContainer<ScalarT> expA;
  Intrepid2::FieldContainer<ScalarT> Fpinv;
  Intrepid2::FieldContainer<ScalarT> FpinvT;
  Intrepid2::FieldContainer<ScalarT> Cpinv;

  Intrepid2::FieldContainer<ScalarT> tmp;
  Intrepid2::FieldContainer<ScalarT> tmp2;

};
}

#endif
