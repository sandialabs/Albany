//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(HeatEqnResidual_hpp)
#define HeatEqnResidual_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

///
/// Heat equation residual evaluator for ACE-LCM
///
template <typename EvalT, typename Traits>
class HeatEqnResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ///
  /// Constructor
  ///
  HeatEqnResidual(
      const Teuchos::ParameterList&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits> &vm);

  ///
  /// Calculates the heat equation residual
  ///
  void
  evaluateFields(typename Traits::EvalData d);

  // update functions:
  void
  updateTemperatureChange(std::size_t cell, std::size_t qp);
  
  void
  update_dfdT(std::size_t cell, std::size_t qp);
  
  void
  updateSaturations(std::size_t cell, std::size_t qp);
  
  // calculation functions:
  ScalarT
  evaluateFreezingCurve(std::size_t cell, std::size_t qp);

private:
  // Input (MDFields):
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<const ScalarT, Cell, QuadPoint> Temperature;
  PHX::MDField<const ScalarT, Cell, QuadPoint> Tdot;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> TGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint> density_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> heat_capacity_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> melting_temperature_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> pressure_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> porosity_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> thermal_conductivity_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> thermal_inertia_;
  
  // Input (ScalarTs):
  ScalarT rho_ice_;
  ScalarT latent_heat_;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TResidual;

  // Workspace:
  unsigned int numQPs, numDims, numNodes, worksetSize;
  Kokkos::DynRankView<ScalarT, PHX::Device> heat_flux_;
  Kokkos::DynRankView<ScalarT, PHX::Device> accumulation_;
  Kokkos::DynRankView<ScalarT, PHX::Device> Temperature_old_;
  Kokkos::DynRankView<ScalarT, PHX::Device> delTemp_;
  Kokkos::DynRankView<ScalarT, PHX::Device> dfdT_;
  Kokkos::DynRankView<ScalarT, PHX::Device> f_;
  Kokkos::DynRankView<ScalarT, PHX::Device> w_;
  Kokkos::DynRankView<ScalarT, PHX::Device> f_old_;
};

} // namespace LCM

#endif // HeatEqnResidual_hpp
