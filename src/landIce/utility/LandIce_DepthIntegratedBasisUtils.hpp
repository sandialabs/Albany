//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_DEPTH_INTEGRATED_BASIS_UTILS_HPP
#define LANDICE_DEPTH_INTEGRATED_BASIS_UTILS_HPP

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_BasicTopologies.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_TestForException.hpp"

#include <Kokkos_DynRankView.hpp>

#include <array>
#include <algorithm>
#include <vector>

namespace LandIce {

/*
 * Utilities for the depth-integrated model, whose FE basis on a (one-layer) wedge is the tensor product
 * of a triangular basis and a 1D vertical basis. In a column, the velocity is
 *
 *   u(zeta) = phiBed(zeta) * u_bed + phiTop(zeta) * u_top,     zeta in [0,1] (0 at the bed, 1 at the top),
 *
 * where phiBed and phiTop are the basis functions of the bottom and top vertices restricted to the vertical
 * edge joining them. These utilities evaluate phiBed/phiTop from the wedge basis itself, so that all the
 * code that needs the vertical profile (vertical averages, export of the velocity at other levels, ...)
 * stays consistent with the basis used in the solve, rather than hard-coding its shape.
 */

namespace depth_integrated_impl {

template<typename DeviceType, typename ValueType>
void checkWedgeBasis (const Intrepid2::Basis<DeviceType,ValueType,ValueType>& wedgeBasis)
{
  TEUCHOS_TEST_FOR_EXCEPTION (wedgeBasis.getBaseCellTopology().getBaseKey()!=shards::Wedge<6>::key, std::logic_error,
      "Error! [LandIce depth-integrated basis utils] Basis '" << wedgeBasis.getName()
      << "' is not defined on a wedge.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (wedgeBasis.getDofCount(0,0)!=1 || wedgeBasis.getDofCount(0,3)!=1, std::logic_error,
      "Error! [LandIce depth-integrated basis utils] Basis '" << wedgeBasis.getName()
      << "' must have exactly one DOF on wedge vertices 0 and 3.\n");
}

} // namespace depth_integrated_impl

// Values of phiBed and phiTop at the normalized heights 'zeta' (in [0,1], 0 at the bed).
// They are the basis functions of wedge vertices 0 (bed) and 3 (top), evaluated on the vertical
// edge of the reference wedge joining them, i.e., at points (0,0,2*zeta-1).
template<typename DeviceType, typename ValueType>
void evalDepthIntegratedVerticalProfile (const Intrepid2::Basis<DeviceType,ValueType,ValueType>& wedgeBasis,
                                         const std::vector<double>& zeta,
                                         std::vector<double>& phiBed,
                                         std::vector<double>& phiTop)
{
  depth_integrated_impl::checkWedgeBasis(wedgeBasis);

  const int numLevels = zeta.size();
  phiBed.resize(numLevels);
  phiTop.resize(numLevels);
  if (numLevels==0) return;

  Kokkos::DynRankView<ValueType,DeviceType> points("points", numLevels, 3);
  auto points_h = Kokkos::create_mirror_view(points);
  for (int il=0; il<numLevels; ++il) {
    points_h(il,0) = 0.0;
    points_h(il,1) = 0.0;
    points_h(il,2) = 2.0*zeta[il] - 1.0;  // reference wedge: z in [-1,1]
  }
  Kokkos::deep_copy(points, points_h);

  Kokkos::DynRankView<ValueType,DeviceType> values("values", wedgeBasis.getCardinality(), numLevels);
  wedgeBasis.getValues(values, points, Intrepid2::OPERATOR_VALUE);
  auto values_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);

  const int bedOrd = wedgeBasis.getDofOrdinal(0, 0, 0);  // DOF of wedge vertex 0
  const int topOrd = wedgeBasis.getDofOrdinal(0, 3, 0);  // DOF of wedge vertex 3
  for (int il=0; il<numLevels; ++il) {
    phiBed[il] = values_h(bedOrd, il);
    phiTop[il] = values_h(topOrd, il);
  }
}

// Vertical averages of phiBed and phiTop over the column: {int_0^1 phiBed dzeta, int_0^1 phiTop dzeta}.
// The vertically averaged velocity is then wBed*u_bed + wTop*u_top. The integrals are computed with a
// Gauss rule exact for polynomials of the basis degree. For the default P1 x I4 basis they are {1/5, 4/5};
// for a basis linear in the vertical direction they are {1/2, 1/2}.
template<typename DeviceType, typename ValueType>
std::array<double,2> computeDepthIntegratedVerticalAverageWeights (const Intrepid2::Basis<DeviceType,ValueType,ValueType>& wedgeBasis)
{
  depth_integrated_impl::checkWedgeBasis(wedgeBasis);

  const shards::CellTopology line(shards::getCellTopologyData<shards::Line<2>>());
  Intrepid2::DefaultCubatureFactory cubFactory;
  auto cubature = cubFactory.create<DeviceType,ValueType,ValueType>(line, std::max<int>(wedgeBasis.getDegree(),1));
  const int numPoints = cubature->getNumPoints();
  Kokkos::DynRankView<ValueType,DeviceType> points("points", numPoints, 1);
  Kokkos::DynRankView<ValueType,DeviceType> weights("weights", numPoints);
  cubature->getCubature(points, weights);
  auto points_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), points);
  auto weights_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weights);

  std::vector<double> zeta(numPoints);
  for (int qp=0; qp<numPoints; ++qp) {
    zeta[qp] = (points_h(qp,0) + 1.0)/2.0;  // [-1,1] -> [0,1]
  }
  std::vector<double> phiBed, phiTop;
  evalDepthIntegratedVerticalProfile(wedgeBasis, zeta, phiBed, phiTop);

  std::array<double,2> w = {0.0, 0.0};
  for (int qp=0; qp<numPoints; ++qp) {
    w[0] += 0.5*weights_h(qp)*phiBed[qp];  // the Jacobian of [-1,1] -> [0,1] is 1/2
    w[1] += 0.5*weights_h(qp)*phiTop[qp];
  }
  return w;
}

} // namespace LandIce

#endif // LANDICE_DEPTH_INTEGRATED_BASIS_UTILS_HPP
