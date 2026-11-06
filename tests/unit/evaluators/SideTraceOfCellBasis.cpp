//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests for PHAL::computeSideTraceOfCellBasis, used by PHAL::ComputeBasisFunctionsSide
// when BF/GradBF on a side are the trace of a cell FE basis (e.g., the LandIce depth-integrated model).

#include "PHAL_SideTraceOfCellBasis.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_DerivedBasis_HGRAD_WEDGE.hpp"
#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C2_FEM.hpp"
#ifdef ALBANY_LANDICE
#include "Intrepid2_HGRAD_LINE_I4_FEM.hpp"
#endif

#include "Teuchos_UnitTestHarness.hpp"

#include <cmath>

namespace {

using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
using Real       = double;
using View       = Kokkos::DynRankView<Real,DeviceType>;
using Basis      = Intrepid2::Basis<DeviceType,Real,Real>;

constexpr double tol = 1e-13;

void getSideCubature (const shards::CellTopology& sideTopo, const int degree, View& points, View& weights)
{
  Intrepid2::DefaultCubatureFactory factory;
  auto cub = factory.create<DeviceType,Real,Real>(sideTopo, degree);
  points  = View("points",  cub->getNumPoints(), cub->getDimension());
  weights = View("weights", cub->getNumPoints());
  cub->getCubature(points, weights);
}

template<typename HostView>
bool valAllZero (const HostView& v, const int side)
{
  for (size_t i=0; i<v.extent(1); ++i)
    for (size_t qp=0; qp<v.extent(2); ++qp)
      if (v(side,i,qp)!=0.0) return false;
  return true;
}

template<typename HostView>
bool gradAllZero (const HostView& g, const int side)
{
  for (size_t i=0; i<g.extent(1); ++i)
    for (size_t qp=0; qp<g.extent(2); ++qp)
      for (size_t d=0; d<g.extent(3); ++d)
        if (g(side,i,qp,d)!=0.0) return false;
  return true;
}

// Check that the trace of 'cellBasis' on the sides of topology 'sideTopo' coincides with 'sideBasis',
// with side nodes ordered as in the Shards side node map. Sides with a different topology must be zero.
void checkTraceEqualsSideBasis (const Basis& cellBasis, const shards::CellTopology& cellTopo,
                                const Basis& sideBasis, const int cubDegree,
                                Teuchos::FancyOStream& out, bool& success)
{
  const auto sideTopo = sideBasis.getBaseCellTopology();
  const int sideDim = sideTopo.getDimension();
  View points, weights;
  getSideCubature(sideTopo, cubDegree, points, weights);
  const int numQPs = points.extent(0);
  const int numSideNodes = sideBasis.getCardinality();

  View sideVals ("sideVals",  numSideNodes, numQPs);
  View sideGrads("sideGrads", numSideNodes, numQPs, sideDim);
  sideBasis.getValues(sideVals,  points, Intrepid2::OPERATOR_VALUE);
  sideBasis.getValues(sideGrads, points, Intrepid2::OPERATOR_GRAD);
  auto sideVals_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sideVals);
  auto sideGrads_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sideGrads);

  View val, grad;
  PHAL::computeSideTraceOfCellBasis(cellBasis, cellTopo, sideTopo, points, val, grad);
  auto val_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
  auto grad_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);

  TEST_EQUALITY (static_cast<int>(val_h.extent(0)), static_cast<int>(cellTopo.getSideCount()));
  TEST_EQUALITY (static_cast<int>(val_h.extent(1)), numSideNodes);
  TEST_EQUALITY (static_cast<int>(val_h.extent(2)), numQPs);
  TEST_EQUALITY (static_cast<int>(grad_h.extent(3)), sideDim);

  for (unsigned side=0; side<cellTopo.getSideCount(); ++side) {
    const shards::CellTopology thisSideTopo(cellTopo.getCellTopologyData(sideDim,side));
    if (thisSideTopo.getBaseKey()!=sideTopo.getBaseKey()) {
      TEST_ASSERT (valAllZero(val_h,side));
      TEST_ASSERT (gradAllZero(grad_h,side));
      continue;
    }
    for (int i=0; i<numSideNodes; ++i) {
      for (int qp=0; qp<numQPs; ++qp) {
        TEST_COMPARE (std::abs(val_h(side,i,qp)-sideVals_h(i,qp)), <=, tol);
        for (int d=0; d<sideDim; ++d) {
          TEST_COMPARE (std::abs(grad_h(side,i,qp,d)-sideGrads_h(i,qp,d)), <=, tol);
        }
      }
    }
  }
}

} // anonymous namespace

// The trace of the trilinear wedge basis must reproduce exactly what ComputeBasisFunctionsSide
// does without a cell FE basis, i.e., bilinear quads on lateral faces and linear triangles on
// the top/bottom faces, with side nodes in Shards order.
TEUCHOS_UNIT_TEST(SideTraceOfCellBasis, WedgeC1)
{
  const shards::CellTopology wedge(shards::getCellTopologyData<shards::Wedge<6>>());
  Intrepid2::Basis_HGRAD_WEDGE_C1_FEM<DeviceType,Real,Real> cellBasis;
  Intrepid2::Basis_HGRAD_QUAD_C1_FEM<DeviceType,Real,Real> quadBasis;
  Intrepid2::Basis_HGRAD_TRI_C1_FEM<DeviceType,Real,Real>  triBasis;

  checkTraceEqualsSideBasis(cellBasis, wedge, quadBasis, 3, out, success);
  checkTraceEqualsSideBasis(cellBasis, wedge, triBasis,  3, out, success);
}

// Same as above, but with the tensor-product (derived) wedge basis, whose DOF ordinals
// follow the tensor structure; the trace must map them to cell vertices through the DOF tags.
TEUCHOS_UNIT_TEST(SideTraceOfCellBasis, DerivedWedgeC1)
{
  using tri_basis  = Intrepid2::Basis_HGRAD_TRI_Cn_FEM<DeviceType,Real,Real>;
  using line_basis = Intrepid2::Basis_HGRAD_LINE_Cn_FEM<DeviceType,Real,Real>;

  const shards::CellTopology wedge(shards::getCellTopologyData<shards::Wedge<6>>());
  Intrepid2::Basis_Derived_HGRAD_WEDGE<tri_basis,line_basis> cellBasis(1);
  Intrepid2::Basis_HGRAD_QUAD_C1_FEM<DeviceType,Real,Real> quadBasis;
  Intrepid2::Basis_HGRAD_TRI_C1_FEM<DeviceType,Real,Real>  triBasis;

  checkTraceEqualsSideBasis(cellBasis, wedge, quadBasis, 3, out, success);
  checkTraceEqualsSideBasis(cellBasis, wedge, triBasis,  3, out, success);
}

// 2D cells: sides are edges (side dimension 1)
TEUCHOS_UNIT_TEST(SideTraceOfCellBasis, QuadAndTriC1)
{
  const shards::CellTopology quad(shards::getCellTopologyData<shards::Quadrilateral<4>>());
  const shards::CellTopology tri (shards::getCellTopologyData<shards::Triangle<3>>());
  Intrepid2::Basis_HGRAD_QUAD_C1_FEM<DeviceType,Real,Real> quadBasis;
  Intrepid2::Basis_HGRAD_TRI_C1_FEM<DeviceType,Real,Real>  triBasis;
  Intrepid2::Basis_HGRAD_LINE_C1_FEM<DeviceType,Real,Real> lineBasis;

  checkTraceEqualsSideBasis(quadBasis, quad, lineBasis, 3, out, success);
  checkTraceEqualsSideBasis(triBasis,  tri,  lineBasis, 3, out, success);
}

// Only bases with exactly one DOF per vertex are supported
TEUCHOS_UNIT_TEST(SideTraceOfCellBasis, UnsupportedBasis)
{
  const shards::CellTopology wedge(shards::getCellTopologyData<shards::Wedge<6>>());
  Intrepid2::Basis_HGRAD_WEDGE_C2_FEM<DeviceType,Real,Real> cellBasis;
  Intrepid2::Basis_HGRAD_QUAD_C1_FEM<DeviceType,Real,Real> quadBasis;
  View points, weights, val, grad;
  getSideCubature(quadBasis.getBaseCellTopology(), 3, points, weights);
  TEST_THROW (PHAL::computeSideTraceOfCellBasis(cellBasis, wedge, quadBasis.getBaseCellTopology(), points, val, grad),
              std::logic_error);
}

#ifdef ALBANY_LANDICE
// Depth-integrated LandIce basis: P1 triangles x 1D quartic (I4) along the vertical.
// On lateral faces, the trace must be (linear along the edge) x (quartic along the vertical),
// regardless of which side reference coordinate is vertical: for Shards Wedge<6>, faces 0=(0,1,4,3)
// and 1=(1,2,5,4) are vertical along the second coordinate, face 2=(0,3,5,2) along the first.
TEUCHOS_UNIT_TEST(SideTraceOfCellBasis, DepthIntegratedWedge)
{
  using tri_basis  = Intrepid2::Basis_HGRAD_TRI_Cn_FEM<DeviceType,Real,Real>;
  using line_basis = Intrepid2::Basis_HGRAD_LINE_I4_FEM<DeviceType,Real,Real>;

  const shards::CellTopology wedge(shards::getCellTopologyData<shards::Wedge<6>>());
  Intrepid2::Basis_Derived_HGRAD_WEDGE<tri_basis,line_basis> cellBasis(1);

  // Top and bottom faces: the trace is the P1 triangle basis
  Intrepid2::Basis_HGRAD_TRI_C1_FEM<DeviceType,Real,Real>  triBasis;
  checkTraceEqualsSideBasis(cellBasis, wedge, triBasis, 3, out, success);

  // Lateral faces
  const shards::CellTopology quad(shards::getCellTopologyData<shards::Quadrilateral<4>>());
  const int numSideNodes = 4;
  const double quadNodes[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};  // Shards/Intrepid2 reference quad
  const int vertDir[3] = {1,1,0};                                // vertical reference direction on faces 0,1,2

  // Vertical shape functions as functions of the vertical reference coordinate v in [-1,1]
  auto psi  = [](const bool bottom, const double v) { const double z=(1-v)/2; return bottom ? std::pow(z,4) : 1-std::pow(z,4); };
  auto dpsi = [](const bool bottom, const double v) { const double z=(1-v)/2; return bottom ? -2*std::pow(z,3) : 2*std::pow(z,3); };

  for (const int cubDegree : {3,4,6}) {
    View points, weights, val, grad;
    getSideCubature(quad, cubDegree, points, weights);
    PHAL::computeSideTraceOfCellBasis(cellBasis, wedge, quad, points, val, grad);
    auto points_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), points);
    auto weights_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weights);
    auto val_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
    auto grad_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    const int numQPs = points_h.extent(0);

    for (int side=0; side<3; ++side) {
      const int vd = vertDir[side];
      const int hd = 1-vd;
      for (int i=0; i<numSideNodes; ++i) {
        const bool bottom = wedge.getNodeMap(2,side,i) < 3;
        // Sanity check of the face orientation: bottom nodes sit at v=-1
        TEST_EQUALITY (quadNodes[i][vd], bottom ? -1.0 : 1.0);

        double integral = 0;
        for (int qp=0; qp<numQPs; ++qp) {
          const double h = points_h(qp,hd);
          const double v = points_h(qp,vd);
          const double lin = (1+quadNodes[i][hd]*h)/2;
          const double expected_val = lin*psi(bottom,v);
          double expected_grad[2];
          expected_grad[hd] = quadNodes[i][hd]/2*psi(bottom,v);
          expected_grad[vd] = lin*dpsi(bottom,v);

          TEST_COMPARE (std::abs(val_h(side,i,qp)-expected_val), <=, tol);
          TEST_COMPARE (std::abs(grad_h(side,i,qp,0)-expected_grad[0]), <=, tol);
          TEST_COMPARE (std::abs(grad_h(side,i,qp,1)-expected_grad[1]), <=, tol);
          integral += weights_h(qp)*val_h(side,i,qp);
        }

        // Integral over the reference face: int_{-1}^{1} lin dh = 1, int_{-1}^{1} psi dv = 2/5 (bottom) or 8/5 (top).
        // These are what split a depth-averaged lateral stress 1/5 (bed) - 4/5 (surface) between the two DOFs,
        // instead of 1/2 - 1/2 with the bilinear basis. Exact only if the cubature degree is at least 4.
        const double exact = bottom ? 2.0/5 : 8.0/5;
        if (cubDegree>=4) {
          TEST_FLOATING_EQUALITY (integral, exact, tol);
        } else {
          TEST_ASSERT (std::abs(integral-exact) > 1e-3);
        }
      }
    }
  }
}
#endif
