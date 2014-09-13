//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef DAMAGE_SOURCE_HPP
#define DAMAGE_SOURCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace LCM {
/** 
 * \brief Damage Source
 */

template<typename EvalT, typename Traits>
class DamageSource : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {
  
public:
  DamageSource(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint> bulkModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> dp;
  PHX::MDField<ScalarT,Cell,QuadPoint> seff;
  PHX::MDField<ScalarT,Cell,QuadPoint> energy;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint> damageLS;
  RealType gc;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> source;

  std::string sourceName;
  std::string damageName;
  unsigned int numQPs;
  unsigned int numDims;
};
}

#endif
