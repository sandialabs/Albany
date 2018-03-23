//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEdensity_hpp)
#define ACEdensity_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates mass density at integration points
///
template <typename EvalT, typename Traits>
class ACEdensity : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>,
                   public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
 public:
   
  using ScalarT          = typename EvalT::ScalarT;

  ACEdensity(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  /// Calculates mixture model density
  void
  evaluateFields(typename Traits::EvalData workset);

  /// Gets the intrinsic density values
  ScalarT&
  getValue(const std::string& n);

 private:
  
  int num_qps_{0};
  int num_dims_{0};
  
  // dependent MDFields
  PHX::MDField<ScalarT, Cell, QuadPoint> porosity_;
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_saturation_;
  PHX::MDField<ScalarT, Cell, QuadPoint> water_saturation_;
  

  // contains the mixture model density value
  PHX::MDField<ScalarT, Cell, QuadPoint> density_;
  
  // contains the intrinsic density values for ice, water, sediment
  // these values are constant
  ScalarT rho_ice_{0.0};
  ScalarT rho_wat_{0.0};
  ScalarT rho_sed_{0.0};

};
}  // namespace LCM

#endif  // ACEdensity_hpp
