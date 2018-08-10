//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACETemperatureResidual_hpp)
#define ACETemperatureResidual_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

///
/// Heat equation residual evaluator for ACE-LCM
///
template <typename EvalT, typename Traits>
class ACETemperatureResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ///
  /// Constructor
  ///
  ACETemperatureResidual(
      Teuchos::ParameterList const&        p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Calculates the heat equation residual
  ///
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  // Input (MDFields):
  PHX::MDField<MeshScalarT const, Cell, Node, QuadPoint>      wbf_;
  PHX::MDField<MeshScalarT const, Cell, Node, QuadPoint, Dim> wgradbf_;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                temperature_;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                tdot_;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           tgrad_;
  PHX::MDField<ScalarT const, Cell, QuadPoint> thermal_conductivity_;
  PHX::MDField<ScalarT const, Cell, QuadPoint> thermal_inertia_;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> residual_;

  // Workspace:
  unsigned int num_qp_, num_dims_, num_nodes_, workset_size_;
  Kokkos::DynRankView<ScalarT, PHX::Device> heat_flux_;
  Kokkos::DynRankView<ScalarT, PHX::Device> accumulation_;
};

}  // namespace LCM

#endif  // ACETemperatureResidual_hpp
