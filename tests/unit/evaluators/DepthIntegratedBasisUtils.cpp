//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests for the LandIce depth-integrated basis utilities, which compute the vertical
// profile and the vertical-average weights of the bed/top basis functions from the wedge basis.

#include "LandIce_DepthIntegratedBasisUtils.hpp"

#include "Intrepid2_DerivedBasis_HGRAD_WEDGE.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_I4_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C1_FEM.hpp"

#include "Teuchos_UnitTestHarness.hpp"

#include <cmath>
#include <functional>
#include <vector>

namespace {

using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
using Real       = double;
using Basis      = Intrepid2::Basis<DeviceType,Real,Real>;

constexpr double tol = 1e-14;

void checkProfileAndWeights (const Basis& basis,
                             const std::function<double(double)>& expectedBed,
                             const std::function<double(double)>& expectedTop,
                             const double expectedWeightBed, const double expectedWeightTop,
                             Teuchos::FancyOStream& out, bool& success)
{
  std::vector<double> zeta;
  for (int i=0; i<=10; ++i) zeta.push_back(i/10.0);
  zeta.push_back(0.123);
  zeta.push_back(0.987);

  std::vector<double> phiBed, phiTop;
  LandIce::evalDepthIntegratedVerticalProfile(basis, zeta, phiBed, phiTop);
  TEST_EQUALITY (phiBed.size(), zeta.size());
  TEST_EQUALITY (phiTop.size(), zeta.size());
  for (size_t il=0; il<zeta.size(); ++il) {
    TEST_COMPARE (std::abs(phiBed[il]-expectedBed(zeta[il])), <=, tol);
    TEST_COMPARE (std::abs(phiTop[il]-expectedTop(zeta[il])), <=, tol);
  }

  const auto w = LandIce::computeDepthIntegratedVerticalAverageWeights(basis);
  TEST_COMPARE (std::abs(w[0]-expectedWeightBed), <=, tol);
  TEST_COMPARE (std::abs(w[1]-expectedWeightTop), <=, tol);
}

} // anonymous namespace

// Default depth-integrated basis (as built by StokesFOBase): P1 triangles x I4 along the vertical,
// phiTop = 1-(1-zeta)^4 and phiBed = (1-zeta)^4, whose vertical averages are 4/5 and 1/5.
TEUCHOS_UNIT_TEST(DepthIntegratedBasisUtils, DepthIntegratedWedge)
{
  using tri_basis  = Intrepid2::Basis_HGRAD_TRI_Cn_FEM<DeviceType,Real,Real>;
  using line_basis = Intrepid2::Basis_HGRAD_LINE_I4_FEM<DeviceType,Real,Real>;
  Intrepid2::Basis_Derived_HGRAD_WEDGE<tri_basis,line_basis> basis(1);

  checkProfileAndWeights(basis,
                         [](double z) { return std::pow(1-z,4); },
                         [](double z) { return 1-std::pow(1-z,4); },
                         1.0/5, 4.0/5, out, success);
}

// Bases linear along the vertical: the weights reduce to the trapezoidal rule used for layered meshes
TEUCHOS_UNIT_TEST(DepthIntegratedBasisUtils, LinearWedges)
{
  using tri_basis  = Intrepid2::Basis_HGRAD_TRI_Cn_FEM<DeviceType,Real,Real>;
  using line_basis = Intrepid2::Basis_HGRAD_LINE_Cn_FEM<DeviceType,Real,Real>;
  Intrepid2::Basis_Derived_HGRAD_WEDGE<tri_basis,line_basis> derivedBasis(1);
  Intrepid2::Basis_HGRAD_WEDGE_C1_FEM<DeviceType,Real,Real> c1Basis;

  auto bed = [](double z) { return 1-z; };
  auto top = [](double z) { return z; };
  checkProfileAndWeights(derivedBasis, bed, top, 0.5, 0.5, out, success);
  checkProfileAndWeights(c1Basis,      bed, top, 0.5, 0.5, out, success);
}

TEUCHOS_UNIT_TEST(DepthIntegratedBasisUtils, NonWedgeBasis)
{
  Intrepid2::Basis_HGRAD_QUAD_C1_FEM<DeviceType,Real,Real> basis;
  std::vector<double> zeta = {0.5}, phiBed, phiTop;
  TEST_THROW (LandIce::evalDepthIntegratedVerticalProfile(basis, zeta, phiBed, phiTop), std::logic_error);
  TEST_THROW (LandIce::computeDepthIntegratedVerticalAverageWeights(basis), std::logic_error);
}
