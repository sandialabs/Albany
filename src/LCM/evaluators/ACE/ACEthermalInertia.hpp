//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEthermalInertia_hpp)
#define ACEthermalInertia_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates thermal inertia at integration points
///
template <typename EvalT, typename Traits>
class ACEthermalInertia : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>,
                          public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT = typename EvalT::ScalarT;

  ///
  /// Constructor
  ///
  ACEthermalInertia(
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
  /// Calculates mixture model thermal inertia
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

  // MDFields that thermal inertia depends on
  PHX::MDField<const ScalarT, Cell, QuadPoint> density_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> heat_capacity_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> dfdT_;

  ///
  /// Contains the mixture model thermal inertia value
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> thermal_inertia_;

  ///
  /// Constants needed to calculate thermal inertia
  ///
  ScalarT latent_heat_{334.0};
  ScalarT rho_ice_{920.0};
};
}  // namespace LCM

#endif  // ACEthermalInertia_hpp
