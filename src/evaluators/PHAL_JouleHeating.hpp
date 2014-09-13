//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_JOULEHEATING_HPP
#define PHAL_JOULEHEATING_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace PHAL {

/** 
 * \brief Joule heating source term for ThermoElectrostatics
 */
template<typename EvalT, typename Traits>
class JouleHeating : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  JouleHeating(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);

private:

  std::size_t numQPs;
  std::size_t numDims;

  // Inputs
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> potentialGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> potentialFlux;

  // Outputs
  PHX::MDField<ScalarT,Cell,QuadPoint> jouleHeating;
};
}

#endif
