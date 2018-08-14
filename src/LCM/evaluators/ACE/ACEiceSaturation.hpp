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
                         public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;

  ///
  /// Constructor
  ///
  ACEiceSaturation(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Calculates evolution of ice saturation
  ///
  void
  evaluateFields(typename Traits::EvalData workset);

  ///
  /// Sacado method to access parameters
  ///
  ScalarT&
  getValue(const std::string& n);

 private:
  ///
  /// Number of integration points
  ///
  int num_qps_{0};

  ///
  /// Number of problem dimensions
  ///
  int num_dims_{0};

  ///
  /// Contains the ice saturation values
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_saturation_;

  // MDFields that ice saturation depends on
  PHX::MDField<ScalarT, Cell, QuadPoint> delta_temperature_;
  PHX::MDField<ScalarT, Cell, QuadPoint> dfdT_;

  ///
  /// Contains the initial ice saturation value
  ///
  ScalarT ice_saturation_init_{0.95};

  ///
  /// Contains the maximum ice saturation value
  ///
  ScalarT max_ice_saturation_{0.95};

  ///
  /// Contains the ice saturation from last timestep
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_saturation_old_;
};
}  // namespace LCM

#endif  // ACEiceSaturation_hpp
