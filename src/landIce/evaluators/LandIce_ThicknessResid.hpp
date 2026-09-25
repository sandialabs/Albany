//*****************************************************************//
//    Albany 3.0: Copyright 2016 Sandia Corporation                  //
//    This software is released under the BSD license described    //
//    in the top-level Albany license.txt.                           //
//*****************************************************************//

#ifndef LANDICE_THICKNESS_RESID_HPP
#define LANDICE_THICKNESS_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"

namespace LandIce {

/**
 * Conservative P1 thickness transport on horizontal triangles.  The triangle
 * can be either a 2D triangular cell or a selected triangular face of a 3D wedge.
 *
 * Available stabilizations: None, SUPG, Graph Viscosity, Edge Stabilization.
 * Inflow thickness defaults to zero and can optionally be specified via
 * the parameter "Inflow Thickness" (km).
 */
template<typename EvalT, typename Traits>
class ThicknessResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits> {
public:

  ThicknessResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);
  void evaluateFields(typename Traits::EvalData workset);

private:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using ParamScalarT= typename EvalT::ParamScalarT;

  PHX::MDField<const ScalarT, Cell, Node> Hdiff; //[km]
  PHX::MDField<const ScalarT, Cell, Node> dHdt;  //[km/yr]
  PHX::MDField<const ParamScalarT, Cell, Node> H0;  //[km]
  PHX::MDField<const RealType, Cell, Node> forcing; //[m/yr]
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> coordVec; //[km]
  PHX::MDField<const RealType> V_cell; // [m/yr],  prescribed: (cell, node, xy)
  PHX::MDField<const ScalarT> V_side; // [m/yr], coupled: (side entry, side node, xy)
  PHX::MDField<ScalarT, Cell, Node> Residual;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<double> dt;
  std::string sideSetName, lateralSideSetName;
  unsigned int cellDim = 0, cubatureDegree = 0;
  bool unsteady = false, lump_mass = false;
  bool supg = false, graph_viscosity = false, edge_stabilization = false;
  RealType inflowThickness = 0.0;

  // Reference-element integration: built just once per evaluator instance.
  Kokkos::DynRankView<RealType,PHX::Device> triBasisValues;
  Kokkos::DynRankView<RealType,PHX::Device> triWeights;
  // Intrepid2 reference P1 triangle gradients and values are computed once.
  // Shape of triRefGrad: (3 basis functions, nqp, 2 reference dimensions).
  Kokkos::DynRankView<RealType, PHX::Device> triRefGrad;

  // Triangle basis evaluated at line cubature mapped to each reference edge.
  Kokkos::DynRankView<RealType,PHX::Device> edgesBasisValues;
  Kokkos::DynRankView<RealType,PHX::Device> edgeWeights;

  // Precomputed for wedge topology.  Entries are parent-cell edge ordinals.
  std::vector<std::vector<int>> edgeSharedByFaces;
  // [triangle face ordinal][parent edge ordinal] -> local triangle edge 0,1,2.
  // For a 2D triangle, face ordinal 0 is an artificial index for the cell.
  std::vector<std::vector<int>> localEdgeForParentEdge;
};

} // namespace LandIce
#endif
