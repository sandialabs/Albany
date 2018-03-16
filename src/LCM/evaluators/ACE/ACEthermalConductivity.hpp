//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEthermalConductivity_hpp)
#define ACEthermalConductivity_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates thermal conductivity at integration points
///
template <typename EvalT, typename Traits>
class ACEthermalConductivity : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>,
                   public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
 public:
  using ScalarT = typename EvalT::ScalarT;

  ACEthermalConductivity(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  /// Calculates mixture model thermal conductivity
  void
  evaluateFields(typename Traits::EvalData workset);

  /// Gets the intrinsic thermal conductivity values
  ScalarT&
  getValue(const std::string& n);

 private:
  int num_qps_{0};
  int num_dims_{0};

  // contains the mixture model thermal conductivity value
  PHX::MDField<ScalarT, Cell, QuadPoint> thermal_conductivity_;

  // contains the intrinsic thermal conductivity values for ice, water, sediment
  // these values are constant
  ScalarT k_ice_{0.0};
  ScalarT k_wat_{0.0};
  ScalarT k_sed_{0.0};

};
}  // namespace LCM

#endif  // ACEthermalConductivity_hpp
