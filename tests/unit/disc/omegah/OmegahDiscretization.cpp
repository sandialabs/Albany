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

#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

template <size_t Dim = 2>
Teuchos::RCP<Albany::OmegahGenericMesh>
createOmegahBoxMesh(const Teuchos::RCP<const Teuchos_Comm>& comm) {
  auto pl = Teuchos::rcp(new Teuchos::ParameterList());
  pl->set("Mesh Creation Method","Box" + std::to_string(Dim) + "D");
  pl->set("Number of Elements",Teuchos::Array<int>(Dim,2));
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

  // Call updateMesh
  disc->updateMesh(); //FIXME hangs in parallel, see screenshot from totalview
  // ~/develop/albanyOmegahAdaptHooks/throwingExceptionInUpdateMesh1d.png

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
