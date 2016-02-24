//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef EQUIVALENT_INCLUSION_CONDUCTIVITY_HPP
#define EQUIVALENT_INCLUSION_CONDUCTIVITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Teuchos_Array.hpp"

namespace LCM {
/** 
 * \brief Evaluates effective thermal conductivity
 */

template<typename EvalT, typename Traits>
class EquivalentInclusionConductivity :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  EquivalentInclusionConductivity(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n);

private:

  int numQPs;
  int numDims;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  // Equivalent inclusion thermal conductivity
  PHX::MDField<ScalarT,Cell,QuadPoint> effectiveK;

  //! Is Young's modulus constant, or random field
  bool is_constant;
  bool isPoroElastic;


  //! Constant value
  ScalarT constant_value;

  //! solid thermal conductivity
  ScalarT condKs;

  //! fluid thermal conductivity
  ScalarT condKf;

  PHX::MDField<ScalarT,Cell,QuadPoint> porosity;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;

#ifdef ALBANY_STOKHOS
  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<RealType>> exp_rf_kl;
#endif

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;
};
}

#endif
