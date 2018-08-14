//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_VECTOR_RESIDUAL_HPP
#define SURFACE_VECTOR_RESIDUAL_HPP

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {

// \brief
//
// Compute the residual forces on a surface
//

template <typename EvalT, typename Traits>
class SurfaceVectorResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SurfaceVectorResidual(
      Teuchos::ParameterList&              p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  /// Length scale parameter for localization zone
  ScalarT thickness_;

  /// Numerical integration rule
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature_;

  /// Finite element basis for the midplane
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
      intrepid_basis_;

  /// First PK Stress
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> stress_;

  /// Current configuration basis
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> current_basis_;

  /// Reference configuration dual basis
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim, Dim> ref_dual_basis_;

  /// Reference configuration normal
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> ref_normal_;

  /// Reference configuration area
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> ref_area_;

  /// Determinant of deformation gradient
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> detF_;

  /// Reference Cell Views
  Kokkos::DynRankView<RealType, PHX::Device> ref_values_;

  Kokkos::DynRankView<RealType, PHX::Device> ref_grads_;

  Kokkos::DynRankView<RealType, PHX::Device> ref_points_;

  Kokkos::DynRankView<RealType, PHX::Device> ref_weights_;

  /// Optional Cohesive Traction
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> traction_;

  // Output:
  /// Force
  PHX::MDField<ScalarT, Cell, Node, Dim> force_;

  /// Cauchy Stress
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> cauchy_stress_;

  unsigned int workset_size_;

  unsigned int num_nodes_;

  unsigned int num_qps_;

  unsigned int num_dims_;

  unsigned int num_surf_nodes_;

  unsigned int num_plane_dims_;

  /// Cohesive Flag
  bool use_cohesive_traction_;

  /// Membrane Forces Flag
  bool compute_membrane_forces_;

  /// Topology modification for adaptive insertion flag.
  bool have_topmod_adaptation_;
};
}  // namespace LCM

#endif
