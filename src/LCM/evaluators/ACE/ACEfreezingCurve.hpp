//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEtemperatureChange_hpp)
#define ACEfreezingCurve_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates the freezing curve at integration points
///
template <typename EvalT, typename Traits>
class ACEfreezingCurve : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>,
                         public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;

  ///
  /// Constructor
  ///
  ACEfreezingCurve(
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
  /// Calculates the freezing curve and evaluated ice saturation
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

  // MDField  that aid freezing curve calculations
  PHX::MDField<const ScalarT, Cell, QuadPoint> Temperature;
  PHX::MDField<const ScalarT, Cell, QuadPoint> melting_temperature_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> delta_temperature_;

  ///
  /// Contains the evaluated ice saturation
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_saturation_evaluated_;

  ///
  /// Contains the evaluated freezing curve slope
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> dfdT_;

  ///
  /// Temperature range over which phase change can occur
  ///
  ScalarT temperature_range_{1.0};
};
}  // namespace LCM

#endif  // ACEfreezingCurve_hpp
