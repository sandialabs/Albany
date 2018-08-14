//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEtemperatureChange_hpp)
#define ACEtemperatureChange_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates the temperature change at integration points
///
template <typename EvalT, typename Traits>
class ACEtemperatureChange : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>,
                             public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;

  ///
  /// Constructor
  ///
  ACEtemperatureChange(
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
  /// Calculates temperature change since last timestep
  ///
  void
  evaluateFields(typename Traits::EvalData workset);

 private:
  ///
  /// Number of integration points
  ///
  int num_qps_{0};

  ///
  /// Number of problem dimensions
  ///
  int num_dims_{0};

  // MDField  that aid temperature change calculation
  PHX::MDField<const ScalarT, Cell, QuadPoint> Temperature;
  PHX::MDField<ScalarT, Cell, QuadPoint>       temperature_old_;
  PHX::MDField<bool, Cell, QuadPoint>          temp_increasing_;
  PHX::MDField<bool, Cell, QuadPoint>          temp_decreasing_;

  ///
  /// Contains the temperature change
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> delta_temperature_{0.0};
};
}  // namespace LCM

#endif  // ACEtemperatureChange_hpp
