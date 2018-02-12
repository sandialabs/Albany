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
/// Heat equation residual evaluator for LCM
///
template <typename EvalT, typename Traits>
class HeatEqnResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  HeatEqnResidual(const Teuchos::ParameterList &p);

  void
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits> &vm);

  void
  evaluateFields(typename Traits::EvalData d);

  // update functions:
  void
  updateMeltingTemperature(std::size_t cell, std::size_t qp);
  
  void
  update_dfdT(std::size_t cell, std::size_t qp);
  
  void
  updateSaturations(std::size_t cell, std::size_t qp);
  
  // calculation functions:
  ScalarT
  thermalConductivity(std::size_t cell, std::size_t qp);
  
  ScalarT
  density(std::size_t cell, std::size_t qp);
  
  ScalarT
  specificHeat(std::size_t cell, std::size_t qp);

  ScalarT
  thermalInertia(std::size_t cell, std::size_t qp);

private:
  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<const ScalarT, Cell, QuadPoint> Temperature;
  PHX::MDField<const ScalarT, Cell, QuadPoint> Tdot;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> TGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint> pressure_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> salinity_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> porosity_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> thermal_K_ice_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> thermal_K_water_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> thermal_K_sed_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> cp_ice_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> cp_water_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> cp_sed_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> rho_ice_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> rho_water_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> rho_sed_;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TResidual;

  // Workspace:
  unsigned int numQPs, numDims, numNodes, worksetSize;
  Kokkos::DynRankView<ScalarT, PHX::Device> heat_flux_;
  Kokkos::DynRankView<ScalarT, PHX::Device> accumulation_;
  Kokkos::DynRankView<ScalarT, PHX::Device> Tmelt_;
  Kokkos::DynRankView<ScalarT, PHX::Device> dfdT_;
  Kokkos::DynRankView<ScalarT, PHX::Device> f_;
  Kokkos::DynRankView<ScalarT, PHX::Device> w_;
};

} // namespace LCM

#endif // HeatEqnResidual_hpp
