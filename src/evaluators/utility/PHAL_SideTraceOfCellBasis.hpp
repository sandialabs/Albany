//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SIDE_TRACE_OF_CELL_BASIS_HPP
#define PHAL_SIDE_TRACE_OF_CELL_BASIS_HPP

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_TestForException.hpp"

#include <Kokkos_DynRankView.hpp>

#include <cmath>
#include <vector>

namespace PHAL {

/*
 * Compute the trace of a nodal cell FE basis on the sides of a cell.
 *
 * For every side s of 'cellTopo' whose topology matches 'sideTopo', and every
 * side cubature point xi_q (given in the side reference element), this computes
 *
 *   val (s,i,q)   = phi_{c(s,i)} ( F_s(xi_q) )
 *   grad(s,i,q,d) = d/dxi_d [ phi_{c(s,i)} ( F_s(xi) ) ] at xi_q
 *
 * where F_s is the Intrepid2 parametrization of side s in the reference cell,
 * c(s,i) is the cell node corresponding to the i-th node of side s (Shards side
 * node map), and phi_j is the cell FE basis function associated with cell vertex j.
 * Gradients are taken w.r.t. the side reference coordinates, i.e., they play the
 * same role as the gradients of a side basis evaluated at the side cubature points.
 *
 * Since the side is parametrized through the reference cell, the result does not
 * depend on how the side reference frame is oriented relative to the cell. This
 * matters, e.g., for tensor-product bases on wedges, where the vertical direction
 * is the second reference coordinate on faces 0 and 1 but the first on face 2.
 *
 * Entries for sides whose topology differs from 'sideTopo' (e.g., the triangular
 * faces of a wedge when 'sideTopo' is a quadrilateral) are set to zero.
 *
 * Requirements: the cell basis must have exactly one DOF per cell vertex and no
 * other DOFs, and all basis functions not associated with the nodes of a side
 * must vanish on that side (true for conforming nodal bases). Both conditions are
 * checked, and an exception is thrown if they do not hold.
 *
 * Input:
 *   cellBasis     : nodal FE basis on the parent cell
 *   cellTopo      : topology of the parent cell
 *   sideTopo      : topology of the sides of interest
 *   sideCubPoints : (numQPs, sideDim) cubature points in the side reference element
 * Output (allocated here):
 *   val  : (numSides, numSideNodes, numQPs)
 *   grad : (numSides, numSideNodes, numQPs, sideDim)
 */
template<typename DeviceType, typename ValueType>
void computeSideTraceOfCellBasis (const Intrepid2::Basis<DeviceType,ValueType,ValueType>& cellBasis,
                                  const shards::CellTopology& cellTopo,
                                  const shards::CellTopology& sideTopo,
                                  const Kokkos::DynRankView<ValueType,DeviceType>& sideCubPoints,
                                  Kokkos::DynRankView<ValueType,DeviceType>& val,
                                  Kokkos::DynRankView<ValueType,DeviceType>& grad)
{
  using view_type = Kokkos::DynRankView<ValueType,DeviceType>;

  const int cellDim      = cellTopo.getDimension();
  const int sideDim      = cellDim-1;
  const int numSides     = cellTopo.getSideCount();
  const int numVertices  = cellTopo.getVertexCount();
  const int numSideNodes = sideTopo.getNodeCount();
  const int numQPs       = sideCubPoints.extent(0);
  const int card         = cellBasis.getCardinality();

  TEUCHOS_TEST_FOR_EXCEPTION (static_cast<int>(sideTopo.getDimension())!=sideDim, std::logic_error,
      "Error! [computeSideTraceOfCellBasis] Side topology '" << sideTopo.getName()
      << "' does not have dimension " << sideDim << ".\n");
  TEUCHOS_TEST_FOR_EXCEPTION (sideCubPoints.rank()!=2 || static_cast<int>(sideCubPoints.extent(1))!=sideDim, std::logic_error,
      "Error! [computeSideTraceOfCellBasis] Side cubature points must have dimensions (numQPs, " << sideDim << ").\n");
  TEUCHOS_TEST_FOR_EXCEPTION (cellBasis.getBaseCellTopology().getBaseKey()!=cellTopo.getBaseKey(), std::logic_error,
      "Error! [computeSideTraceOfCellBasis] Cell basis '" << cellBasis.getName() << "' is defined on '"
      << cellBasis.getBaseCellTopology().getName() << "', but the cell topology is '" << cellTopo.getName() << "'.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (card!=numVertices || static_cast<int>(cellTopo.getNodeCount())!=numVertices, std::logic_error,
      "Error! [computeSideTraceOfCellBasis] Only cell bases with exactly one DOF per vertex are supported.\n"
      "       Basis '" << cellBasis.getName() << "' has cardinality " << card << ", cell '"
      << cellTopo.getName() << "' has " << numVertices << " vertices.\n");

  // Basis ordinal associated with each cell vertex (do not assume the basis follows Shards ordering)
  std::vector<int> vertexToBasisOrd(numVertices);
  for (int v=0; v<numVertices; ++v) {
    TEUCHOS_TEST_FOR_EXCEPTION (cellBasis.getDofCount(0,v)!=1, std::logic_error,
        "Error! [computeSideTraceOfCellBasis] Basis '" << cellBasis.getName() << "' does not have exactly one DOF on vertex " << v << ".\n");
    vertexToBasisOrd[v] = cellBasis.getDofOrdinal(0,v,0);
  }

  val  = view_type("side_trace_val",  numSides, numSideNodes, numQPs);
  grad = view_type("side_trace_grad", numSides, numSideNodes, numQPs, sideDim);
  auto val_h  = Kokkos::create_mirror_view(val);
  auto grad_h = Kokkos::create_mirror_view(grad);
  Kokkos::deep_copy(val_h, 0.0);
  Kokkos::deep_copy(grad_h, 0.0);

  // Affine side parametrizations: F_s(xi)[k] = p(s,k,0) + sum_d p(s,k,d+1)*xi[d]
  const auto param   = Intrepid2::RefSubcellParametrization<DeviceType>::get(sideDim, cellTopo.getKey());
  const auto param_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), param);

  view_type refPoints ("refPoints", numQPs, cellDim);
  view_type cellVals  ("cellVals",  card, numQPs);
  view_type cellGrads ("cellGrads", card, numQPs, cellDim);

  constexpr double tol = 1e-12;
  for (int side=0; side<numSides; ++side) {
    const shards::CellTopology thisSideTopo (cellTopo.getCellTopologyData(sideDim,side));
    if (thisSideTopo.getBaseKey()!=sideTopo.getBaseKey()) continue;

    Intrepid2::CellTools<DeviceType>::mapToReferenceSubcell(refPoints, sideCubPoints, sideDim, side, cellTopo);
    cellBasis.getValues(cellVals,  refPoints, Intrepid2::OPERATOR_VALUE);
    cellBasis.getValues(cellGrads, refPoints, Intrepid2::OPERATOR_GRAD);
    auto cellVals_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cellVals);
    auto cellGrads_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cellGrads);

    std::vector<bool> onSide(numVertices,false);
    for (int i=0; i<numSideNodes; ++i) {
      const int cellNode = cellTopo.getNodeMap(sideDim,side,i);
      const int ord = vertexToBasisOrd[cellNode];
      onSide[cellNode] = true;
      for (int qp=0; qp<numQPs; ++qp) {
        val_h(side,i,qp) = cellVals_h(ord,qp);
        // Chain rule: d(phi o F_s)/dxi_d = sum_k dphi/dx_k * dF_s[k]/dxi_d
        for (int d=0; d<sideDim; ++d) {
          ValueType g = 0;
          for (int k=0; k<cellDim; ++k) {
            g += cellGrads_h(ord,qp,k)*param_h(side,k,d+1);
          }
          grad_h(side,i,qp,d) = g;
        }
      }
    }

    // The trace must be fully described by the side nodes
    for (int v=0; v<numVertices; ++v) {
      if (onSide[v]) continue;
      for (int qp=0; qp<numQPs; ++qp) {
        TEUCHOS_TEST_FOR_EXCEPTION (std::abs(cellVals_h(vertexToBasisOrd[v],qp))>tol, std::logic_error,
            "Error! [computeSideTraceOfCellBasis] Basis function of vertex " << v << " of basis '" << cellBasis.getName()
            << "' does not vanish on side " << side << ". The basis is not a conforming nodal basis.\n");
      }
    }
  }

  Kokkos::deep_copy(val, val_h);
  Kokkos::deep_copy(grad, grad_h);
}

} // namespace PHAL

#endif // PHAL_SIDE_TRACE_OF_CELL_BASIS_HPP
