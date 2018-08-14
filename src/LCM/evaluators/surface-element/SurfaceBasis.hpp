//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_BASIS_HPP
#define SURFACE_BASIS_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {

/// \brief Surface Basis Evaluator
///
/// This evaluator computes bases for surface elements
/// \tparam EvalT
/// \tparam Traits
///
template <typename EvalT, typename Traits>
class SurfaceBasis : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  /// \param[in] p Teuchos::ParameterList
  /// \param[in] dl RCP to Albany::Layout
  ///
  SurfaceBasis(
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
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

  ///
  /// Takes given coordinates and computes the corresponding midplane
  /// \param refCoords
  /// \param midplane_coords
  ///
  template <typename ST>
  void
  computeMidplaneCoords(
      PHX::MDField<const ST, Cell, Vertex, Dim> const coords,
      Kokkos::DynRankView<ST, PHX::Device>&           midplane_coords);

  ///
  /// Computes basis from the reference midplane
  /// \param midplane_coords
  /// \param basis
  ///
  template <typename ST>
  void
  computeBasisVectors(
      Kokkos::DynRankView<ST, PHX::Device> const& midplane_coords,
      PHX::MDField<ST, Cell, QuadPoint, Dim, Dim> basis);

  ///
  /// Computes the Dual from the midplane and reference bases
  /// \param midplane_coords
  /// \param basis
  /// \param normal
  /// \param dual_basis
  ///
  void
  computeDualBasisVectors(
      Kokkos::DynRankView<MeshScalarT, PHX::Device> const& midplane_coords,
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> const basis,
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim>            normal,
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim>       dual_basis);

  ///
  /// Computes the jacobian mapping - da/dA
  /// \param basis
  /// \param dual_basis
  /// \param area
  ///
  void
  computeJacobian(
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> const basis,
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> const dual_basis,
      PHX::MDField<MeshScalarT, Cell, QuadPoint>                 area);

 private:
  unsigned int container_size, num_dims_, num_nodes_, num_qps_, num_surf_nodes_,
      num_surf_dims_;

  bool need_current_basis_;

  ///
  /// Input: Cordinates in the reference configuration
  ///
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> reference_coords_;

  ///
  /// Input: Numerical integration rule
  ///
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature_;

  ///
  /// Input: Finite element basis for the midplane
  ///
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
      intrepid_basis_;

  ///
  /// Local View to store the reference midplane_coords
  ///
  Kokkos::DynRankView<MeshScalarT, PHX::Device> ref_midplane_coords_;

  ///
  /// Local View to store the current midplane_coords
  ///
  Kokkos::DynRankView<ScalarT, PHX::Device> current_midplane_coords_;

  ///
  /// Output: Reference basis
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> ref_basis_;

  ///
  /// Output: Reference integration area
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint> ref_area_;

  ///
  /// Output: Reference dual basis
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> ref_dual_basis_;

  ///
  /// Output: Reference normal
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> ref_normal_;

  // if we need to compute the current bases (for mechanics)
  ///
  /// Optional Input: Coordinates in the current configuration
  ///
  PHX::MDField<const ScalarT, Cell, Vertex, Dim> current_coords_;

  ///
  /// Optional Output: Current basis
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> current_basis_;

  ///
  /// Reference Cell View for basis values
  ///
  Kokkos::DynRankView<RealType, PHX::Device> ref_values_;

  ///
  /// Reference Cell View for basis gradients
  ///
  Kokkos::DynRankView<RealType, PHX::Device> ref_grads_;

  ///
  /// Reference Cell View for integration point locations
  ///
  Kokkos::DynRankView<RealType, PHX::Device> ref_points_;

  ///
  /// Reference Cell View for integration weights
  ///
  Kokkos::DynRankView<RealType, PHX::Device> ref_weights_;
};
}  // namespace LCM

#endif  // SURFACE_BASIS_HPP
