//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEwaterSaturation_hpp)
#define ACEwaterSaturation_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates water saturation at integration points
///
template <typename EvalT, typename Traits>
class ACEwaterSaturation : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>,
                           public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;

  ///
  /// Constructor
  ///
  ACEwaterSaturation(
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
  /// Calculates evolution of water saturation
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
  /// Minimum water saturation
  ///
  ScalarT min_water_saturation_{0.05};

  // MDFields that water saturation depends on
  PHX::MDField<const ScalarT, Cell, QuadPoint> ice_saturation_;

  ///
  /// Contains the water saturation values
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> water_saturation_;
};
}  // namespace LCM

#endif  // ACEwaterSaturation_hpp
