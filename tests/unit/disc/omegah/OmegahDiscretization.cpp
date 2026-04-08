//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"
#include "Albany_OmegahDiscretization.hpp"
#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

#include "Shards_CellTopology.hpp"

#include <Omega_h_for.hpp>
#include <Omega_h_array_ops.hpp>

#include <algorithm>

#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

template <size_t Dim = 2>
Teuchos::RCP<Albany::OmegahGenericMesh>
createOmegahBoxMesh(const Teuchos::RCP<const Teuchos_Comm>& comm,
                    const int worksetSize = -1) {
  auto pl = Teuchos::rcp(new Teuchos::ParameterList());
  pl->set("Mesh Creation Method","Box" + std::to_string(Dim) + "D");
  pl->set("Number of Elements",Teuchos::Array<int>(Dim,2));
  if (worksetSize > 0)
    pl->set("Workset Size", worksetSize);
  auto p = Teuchos::rcp(new Albany::OmegahGenericMesh(pl));
  return p;
}

Teuchos::RCP<Albany::OmegahDiscretization>
createOmegahDiscretization(const Teuchos::RCP<Albany::OmegahGenericMesh>& mesh,
                           const Teuchos::RCP<const Teuchos_Comm>& comm,
                           const int neq = 1)
{
  auto discParams = Teuchos::rcp(new Teuchos::ParameterList());
  discParams->set("Number Of Time Derivatives", 0);

  return Teuchos::rcp(new Albany::OmegahDiscretization(
      discParams, neq, mesh, comm));
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_Constructor)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);
  out << "Testing OmegahDiscretization constructor\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_updateMesh_DOFManagers)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);

  // Create solution MFA
  disc->setFieldData();

  // Call updateMesh
  disc->updateMesh();

  // Verify that DOF managers were created
  auto sol_dof_mgr = disc->getDOFManager();
  REQUIRE(Teuchos::nonnull(sol_dof_mgr));

  auto node_dof_mgr = disc->getNodeDOFManager();
  REQUIRE(Teuchos::nonnull(node_dof_mgr));

  // Verify connectivity size matches expected topology (triangles have 3 nodes)
  REQUIRE(3 == sol_dof_mgr->getAlbanyConnManager()->getConnectivitySize(0));
  REQUIRE(3 == node_dof_mgr->getAlbanyConnManager()->getConnectivitySize(0));

  out << "Testing OmegahDiscretization::updateMesh() DOF managers\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_updateMesh_Worksets)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);

  // Create solution MFA
  disc->setFieldData();

  // Call updateMesh
  disc->updateMesh();

  // Get mesh information
  auto ohMesh = mesh->getOmegahMesh();
  auto owned = ohMesh->owned(ohMesh->dim());
  int numOwnedElems = Omega_h::get_sum(owned);

  // Verify workset information
  const auto& wsEBNames = disc->getWsEBNames();
  const auto& wsPhysIndex = disc->getWsPhysIndex();

  REQUIRE(wsEBNames.size() > 0);
  REQUIRE(wsPhysIndex.size() == wsEBNames.size());

  // Verify total elements across all worksets matches owned elements
  int totalWSElems = 0;
  auto ws_sizes = disc->getWorksetsSizes();
  for ( auto& size : ws_sizes ) {
    totalWSElems += size;
  }
  REQUIRE(totalWSElems == numOwnedElems);

  out << "Testing OmegahDiscretization::updateMesh() workset computation\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_updateMesh_Coordinates)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);

  // Create solution MFA
  disc->setFieldData();

  // Call updateMesh
  disc->updateMesh();

  // Verify coordinates were set up
  const auto& coords = disc->getCoordinates();
  REQUIRE(coords.size() > 0);

  // Verify coordinates are dimension * number of nodes
  auto node_dof_mgr = disc->getNodeDOFManager();
  auto node_vs = disc->getOverlapNodeVectorSpace();
  int numNodes = Albany::getLocalSubdim(node_vs);
  int meshDim = disc->getNumDim();

  REQUIRE(coords.size() == meshDim * numNodes);

  out << "Testing OmegahDiscretization::updateMesh() coordinates setup\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_updateMesh_1D)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh<1>(teuchosComm);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);

  // Create solution MFA
  disc->setFieldData();

  // Call updateMesh
  disc->updateMesh();

  // Verify DOF managers were created
  auto sol_dof_mgr = disc->getDOFManager();
  REQUIRE(Teuchos::nonnull(sol_dof_mgr));

  // For 1D (line elements), connectivity size should be 2
  REQUIRE(2 == sol_dof_mgr->getAlbanyConnManager()->getConnectivitySize(0));

  // Verify mesh dimension
  REQUIRE(1 == disc->getNumDim());

  out << "Testing OmegahDiscretization::updateMesh() for 1D mesh\n";
  success = true;
}

// Helper: sum the number of side structs over all worksets for a given sideset name
static int countSideSetEntries(const Albany::OmegahDiscretization& disc,
                               const std::string& sideSetName)
{
  int total = 0;
  int num_ws = disc.getNumWorksets();
  for (int ws = 0; ws < num_ws; ++ws) {
    const auto& ssl = disc.getSideSets(ws);
    auto it = ssl.find(sideSetName);
    if (it != ssl.end())
      total += static_cast<int>(it->second.size());
  }
  return total;
}

// For the 2D box mesh, there are 4 boundary sidesets (one per side of the unit square).
// Each sideset name matches one of SideSet0..SideSet3.
static const std::vector<std::string> kExpected2DBoxSSNames = {
  "SideSet0", "SideSet1", "SideSet2", "SideSet3"
};
// A 2x2 box mesh has 2 boundary edges per side of the unit square.
static constexpr int kEdgesPerBoundarySide = 2;

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_computeSideSets_basic)
{
  auto teuchosComm = Albany::getDefaultComm();

  // A 2D 2x2 box mesh has SideSet0..SideSet3 (one per boundary edge group).
  auto mesh = createOmegahBoxMesh<2>(teuchosComm);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);

  disc->setFieldData();
  disc->updateMesh();  // internally calls computeSideSets()

  const auto& ssNames = mesh->meshSpecs[0]->ssNames;
  REQUIRE(ssNames.size() == kExpected2DBoxSSNames.size());

  // Verify the expected sideset names are all present.
  for (const auto& expected : kExpected2DBoxSSNames) {
    auto it = std::find(ssNames.begin(), ssNames.end(), expected);
    REQUIRE(it != ssNames.end());
  }

  int num_ws = disc->getNumWorksets();
  REQUIRE(num_ws >= 1);

  // Every workset must have an entry for every sideset name (possibly empty).
  for (int ws = 0; ws < num_ws; ++ws) {
    const auto& ssl = disc->getSideSets(ws);
    for (const auto& ssn : ssNames) {
      REQUIRE(ssl.count(ssn) == 1);
    }
  }

  // For a 2D 2x2 simplex box (8 triangles), each of the 4 boundary edges
  // belongs to exactly one SideSet. Count them per-sideset (globally summed).
  // The boundary has 2 edges per side, so each sideset should have 2 entries.
  // When running on multiple MPI ranks the owned-edge sets are disjoint but sum to 2.
  int totalSides = 0;
  for (const auto& ssn : ssNames) {
    int localCount  = countSideSetEntries(*disc, ssn);
    int globalCount = 0;
    Teuchos::reduceAll(*teuchosComm, Teuchos::REDUCE_SUM, 1, &localCount, &globalCount);

    // Each side of the 2x2 box has exactly 2 boundary edges.
    REQUIRE(globalCount == kEdgesPerBoundarySide);
    totalSides += globalCount;
  }
  // 4 sides x 2 edges each = 8 total boundary edges
  REQUIRE(totalSides == static_cast<int>(kExpected2DBoxSSNames.size()) * kEdgesPerBoundarySide);

  // Verify SideStruct fields are in valid ranges for every entry.
  // A triangle has 3 sides, so side_pos must be in [0,2].
  const int num_loc_sides = 3;  // triangle
  for (int ws = 0; ws < num_ws; ++ws) {
    const auto& ssl = disc->getSideSets(ws);
    for (const auto& ssn : ssNames) {
      for (const auto& s : ssl.at(ssn)) {
        REQUIRE(s.elem_GID >= 0);
        REQUIRE(s.side_GID >= 0);
        REQUIRE(s.ws_elem_idx >= 0);
        REQUIRE(s.side_pos  >= 0);
        REQUIRE(s.side_pos   < num_loc_sides);
        REQUIRE(s.elem_ebIndex == 0);
      }
    }
  }

  out << "Testing OmegahDiscretization::computeSideSets() basic correctness\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, Discretization_computeSideSets_multiWorkset)
{
  auto teuchosComm = Albany::getDefaultComm();

  // Force multiple worksets by setting workset size to 1.
  // A 2x2 simplex box mesh has 8 triangles total (globally).
  auto mesh = createOmegahBoxMesh<2>(teuchosComm, 1);
  auto disc = createOmegahDiscretization(mesh, teuchosComm);

  disc->setFieldData();
  disc->updateMesh();

  const auto& ssNames = mesh->meshSpecs[0]->ssNames;
  REQUIRE(ssNames.size() == kExpected2DBoxSSNames.size());

  int num_ws = disc->getNumWorksets();

  // Every workset must contain an entry for every sideset name.
  for (int ws = 0; ws < num_ws; ++ws) {
    const auto& ssl = disc->getSideSets(ws);
    for (const auto& ssn : ssNames) {
      REQUIRE(ssl.count(ssn) == 1);
    }
  }

  // The global side counts should be the same as in the single-workset case.
  int totalSides = 0;
  for (const auto& ssn : ssNames) {
    int localCount  = countSideSetEntries(*disc, ssn);
    int globalCount = 0;
    Teuchos::reduceAll(*teuchosComm, Teuchos::REDUCE_SUM, 1, &localCount, &globalCount);
    REQUIRE(globalCount == kEdgesPerBoundarySide);
    totalSides += globalCount;
  }
  REQUIRE(totalSides == static_cast<int>(kExpected2DBoxSSNames.size()) * kEdgesPerBoundarySide);

  out << "Testing OmegahDiscretization::computeSideSets() with multiple worksets\n";
  success = true;
}
