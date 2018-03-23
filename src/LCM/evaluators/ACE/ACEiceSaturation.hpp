//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEiceSaturation_hpp)
#define ACEiceSaturation_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates ice saturation at integration points
///
template <typename EvalT, typename Traits>
class ACEiceSaturation : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>,
                         public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
 public:
  using ScalarT = typename EvalT::ScalarT;

  ACEiceSaturation(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  /// Calculates evolution of ice saturation
  void
  evaluateFields(typename Traits::EvalData workset);

  /// Gets the saturation values
  //ScalarT&
  //getValue(const std::string& n);

 private:
  int num_qps_{0};
  int num_dims_{0};

  // contains the ice/water saturation values
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_saturation_;
  
  // contains the initial ice saturation value
  ScalarT ice_saturation_init_{1.0};

};
}  // namespace LCM

#endif  // ACEiceSaturation_hpp
