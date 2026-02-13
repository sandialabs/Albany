//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "OmegahGhost.hpp"
#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_Omegah.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

#include <Omega_h_build.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_comm.hpp>
#include <Omega_h_file.hpp> //Omega_h::vtk::write_parallel
#include <Omega_h_inertia.hpp> //Omega_h::inertia::Rib

// Helper function to create 2D simplex box mesh
Omega_h::Mesh createBoxMesh2D(int nx, int ny, bool cutAlongAxis=true) {
  auto lib = Albany::get_omegah_lib();
  const auto scale = 1.0;
  const auto nz = 0;
  auto mesh = Omega_h::build_box(lib.world(),OMEGA_H_SIMPLEX,
                                          scale,scale,scale,
                                          nx,ny,nz);
  // simplify the tests by setting the mesh partition to have cuts along x and y axis 
  auto hints = std::make_shared<Omega_h::inertia::Rib>();
  if(cutAlongAxis) {
    hints->axes.push_back(Omega_h::vector_3(1, 0, 0));  // Cut along x-axis first
    hints->axes.push_back(Omega_h::vector_3(0, 1, 0));  // Then y-axis
  }
  // balance using RIB with the specified axes
  mesh.set_rib_hints(hints);
  mesh.balance();
  // add ghosts
  const auto verbose = false;
  mesh.set_parting(OMEGA_H_GHOSTED, 1, verbose);
  return mesh;
}

// Helper: Create simple 1D mesh with build_from_elems2verts
// Creates a line of 'n_edges' edges: 0--1--2--...--n_edges
Omega_h::Mesh create1DMesh(int n_edges) {
  auto lib = Albany::get_omegah_lib();
  const auto scale = 1.0;
  const auto ny = 0;
  const auto nz = 0;
  auto mesh = Omega_h::build_box(lib.world(),OMEGA_H_SIMPLEX,
                                          scale,scale,scale,
                                          n_edges,ny,nz);
  const auto verbose = false;
  mesh.set_parting(OMEGA_H_GHOSTED, 1, verbose);
  return mesh;
}

//
// TEST 1: Basic ownership count
//
TEUCHOS_UNIT_TEST(OmegahGhost, getNumOwnedElms)
{
  auto mesh = createBoxMesh2D(4, 4);

  auto numOwnedElms = OmegahGhost::getNumOwnedElms(mesh);

  // Each rank should own some elements
  TEST_ASSERT(numOwnedElms > 0);

  // 4x4 quad mesh = 32 triangles (2 per quad)
  int expected = 32;
  int globalCount;
  auto comm = Albany::getDefaultComm();
  Teuchos::reduceAll<int>(*comm, Teuchos::REDUCE_SUM, 1, &numOwnedElms, &globalCount);

  TEST_EQUALITY_CONST(globalCount, expected);
}

//
// TEST 2: Owned entity GIDs
//
TEUCHOS_UNIT_TEST(OmegahGhost, getOwnedEntityGids)
{
  const int ne=4;
  auto mesh = createBoxMesh2D(ne, ne);
  auto comm = Albany::getDefaultComm();

  // Test vertices (dim = 0)
  {
  auto vtxGids = OmegahGhost::getOwnedEntityGids(mesh, 0);
  TEST_ASSERT(vtxGids.size() > 0);
  const int expectedVtx = (ne+1)*(ne+1); //square grid of (ne+1)^2 vertices
  int localCount = vtxGids.size();
  int globalCount;
  Teuchos::reduceAll<int>(*comm, Teuchos::REDUCE_SUM, 1, &localCount, &globalCount);
  TEST_EQUALITY_CONST(globalCount, expectedVtx);
  }

  // Test edges (dim = 1)
  {
  auto edgeGids = OmegahGhost::getOwnedEntityGids(mesh, 1);
  TEST_ASSERT(edgeGids.size() > 0);
  const int expectedEdges = 2 * ((ne+1) * ne) // horizontal + vertical edges
                            + (ne * ne); // diagonal edges
  int localCount = edgeGids.size();
  int globalCount;
  Teuchos::reduceAll<int>(*comm, Teuchos::REDUCE_SUM, 1, &localCount, &globalCount);
  TEST_EQUALITY_CONST(globalCount, expectedEdges);
  }

  // Test triangles (dim = 2)
  {
  auto triGids = OmegahGhost::getOwnedEntityGids(mesh, 2);
  TEST_ASSERT(triGids.size() > 0);
  int expectedTris = ne*ne*2; // each quad is divided into two triangles
  int localCount = triGids.size();
  int globalCount;
  Teuchos::reduceAll<int>(*comm, Teuchos::REDUCE_SUM, 1, &localCount, &globalCount);
  TEST_EQUALITY_CONST(globalCount, expectedTris);
  }
}

//
// TEST 3: Entities in closure - 2d
//
TEUCHOS_UNIT_TEST(OmegahGhost, getEntsInClosureOfOwnedElms)
{
  auto mesh = createBoxMesh2D(4, 4);

  // elements
  auto elmMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, mesh.dim());
  auto numInClosure = Omega_h::get_sum(elmMask);
  auto numOwned = OmegahGhost::getNumOwnedElms(mesh);
  TEST_EQUALITY_CONST(numInClosure, numOwned);

  // vertices and edges
  for(int dim=0; dim<mesh.dim(); dim++) {
    auto entMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, dim);
    auto numOwnedEnts = Omega_h::get_sum(mesh.owned(dim));
    auto numEntsInClosure = Omega_h::get_sum(entMask);
    TEST_ASSERT(numEntsInClosure > 0);
    TEST_ASSERT(numEntsInClosure >= numOwnedEnts);
    TEST_ASSERT(numEntsInClosure <= mesh.nents(dim));
  }
}

//
// TEST 4: GIDs in closure
//
TEUCHOS_UNIT_TEST(OmegahGhost, getEntGidsInClosureOfOwnedElms)
{
  const int n_edges = 8;
  auto mesh = create1DMesh(n_edges);
  auto comm = Albany::getDefaultComm();
  auto numRanks = comm->getSize();
  auto rank = comm->getRank();

  for(int dim=0; dim<=mesh.dim(); dim++) {
    auto gids = OmegahGhost::getEntGidsInClosureOfOwnedElms(mesh, dim);
    //this could be more general for n_edges and numRanks...
    if(numRanks == 4) {
      for(int i=0; i<gids.size(); i++) {
        TEST_ASSERT((rank*2)+i == gids[i]);
      }
    }
    if(numRanks == 1) {
      TEST_ASSERT(mesh.globals(dim) == gids);
    }
  }
}

//
// TEST 5: Vertex coordinates in closure
//
TEUCHOS_UNIT_TEST(OmegahGhost, getVtxCoordsInClosureOfOwnedElms)
{
  auto mesh = createBoxMesh2D(2, 2);

  auto comm = Albany::getDefaultComm();
  auto numRanks = comm->getSize();
  auto rank = comm->getRank();
  auto coords = OmegahGhost::getVtxCoordsInClosureOfOwnedElms(mesh);
  if(numRanks > 1) {
    TEST_ASSERT(numRanks == 4); //hardcoded for four ranks
    auto x = Omega_h::get_component(coords, 2, 0);
    auto minX = Omega_h::get_min(x);
    auto maxX = Omega_h::get_max(x);
    auto y = Omega_h::get_component(coords, 2, 1);
    auto minY = Omega_h::get_min(y);
    auto maxY = Omega_h::get_max(y);
    const auto length = 1.0;
    const auto half = length/2;
    const auto tol = 1e-10;
    if(rank == 0) { // lower left
      TEST_FLOATING_EQUALITY(minX, 0, tol);
      TEST_FLOATING_EQUALITY(maxX, half, tol);
      TEST_FLOATING_EQUALITY(minY, 0, tol);
      TEST_FLOATING_EQUALITY(maxY, half, tol);
    } else if(rank == 1) { // upper left
      TEST_FLOATING_EQUALITY(minX, 0, tol);
      TEST_FLOATING_EQUALITY(maxX, half, tol);
      TEST_FLOATING_EQUALITY(minY, half, tol);
      TEST_FLOATING_EQUALITY(maxY, length, tol);
    } else if(rank == 2) { // lower right
      TEST_FLOATING_EQUALITY(minX, half, tol);
      TEST_FLOATING_EQUALITY(maxX, length, tol);
      TEST_FLOATING_EQUALITY(minY, 0, tol);
      TEST_FLOATING_EQUALITY(maxY, half, tol);
    } else if(rank == 3) { // upper right
      TEST_FLOATING_EQUALITY(minX, half, tol);
      TEST_FLOATING_EQUALITY(maxX, length, tol);
      TEST_FLOATING_EQUALITY(minY, half, tol);
      TEST_FLOATING_EQUALITY(maxY, length, tol);
    }
  } else { // one rank
    TEST_ASSERT(mesh.coords() == coords);
  }
}

//
// TEST 6: Downward Adjacency (Edge→Vertex) for 1D Mesh
//
TEUCHOS_UNIT_TEST(OmegahGhost, getDownAdjacentEntsInClosureOfOwnedElms_1D)
{
  const int n_edges = 8;
  auto mesh = create1DMesh(n_edges);
  auto comm = Albany::getDefaultComm();
  auto numRanks = comm->getSize();
  auto rank = comm->getRank();

  // Get edge-to-vertex downward adjacencies for owned edges
  auto edgeToVtx = OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(mesh, OMEGA_H_VERT);
  auto numOwnedEdges = OmegahGhost::getNumOwnedElms(mesh);

  // Should have 2 vertices per edge
  TEST_EQUALITY_CONST(edgeToVtx.size(), numOwnedEdges * 2);
 
  // Get vertex global ids
  auto vtxGids = mesh.globals(OMEGA_H_VERT);

  if(numRanks > 1) {
    TEST_ASSERT(numRanks == 4); //hardcoded for four ranks
  } 
  // Check connectivity is correct
  auto edgeToVtx_h = Omega_h::HostRead(edgeToVtx);
  for (int e = 0; e < numOwnedEdges; ++e) {
    int v0 = edgeToVtx_h[e * 2 + 0];
    int v1 = edgeToVtx_h[e * 2 + 1];

    // Edge e should connect vertices e and e+1
    TEST_EQUALITY_CONST(vtxGids[v0], (rank*2)+e);
    TEST_EQUALITY_CONST(vtxGids[v1], (rank*2)+e+1);
  }
}

//
// TEST 7: Upward Adjacency (Vertex->Edge) for 1D Mesh
//
TEUCHOS_UNIT_TEST(OmegahGhost, getUpAdjacentEntsInClosureOfOwnedElms_1D_Serial)
{
  auto comm = Albany::getDefaultComm();
  auto numRanks = comm->getSize();
  if( numRanks > 1 ) {
    return;
  }
  // Create 1D mesh with 4 edges: 0--1--2--3--4
  // Vertex adjacencies:
  //   v0: edge 0
  //   v1: edges 0, 1
  //   v2: edges 1, 2
  //   v3: edges 2, 3
  //   v4: edge 3
  const int n_edges = 4;
  auto mesh = create1DMesh(n_edges);

  // Get vertex-to-edge upward adjacencies
  auto vtxToEdge = OmegahGhost::getUpAdjacentEntsInClosureOfOwnedElms(mesh, 0);
  auto numVtxInClosure = OmegahGhost::getNumEntsInClosureOfOwnedElms(mesh, 0);

  // Graph should have entries for vertices in closure
  TEST_EQUALITY_CONST(vtxToEdge.a2ab.size(), numVtxInClosure + 1);

  auto a2ab_h = Omega_h::HostRead(vtxToEdge.a2ab);
  auto ab2b_h = Omega_h::HostRead(vtxToEdge.ab2b);

  // Verify topology:
  // - Boundary vertices (v0, v4): 1 adjacent edge
  // - Interior vertices (v1, v2, v3): 2 adjacent edges

  // Vertex 0: should have 1 edge
  int v0_nedges = a2ab_h[1] - a2ab_h[0];
  TEST_EQUALITY_CONST(v0_nedges, 1);
  TEST_EQUALITY_CONST(ab2b_h[a2ab_h[0]], 0); // edge 0

  // Vertex 1: should have 2 edges
  int v1_nedges = a2ab_h[2] - a2ab_h[1];
  TEST_EQUALITY_CONST(v1_nedges, 2);

  // Vertex 2: should have 2 edges
  int v2_nedges = a2ab_h[3] - a2ab_h[2];
  TEST_EQUALITY_CONST(v2_nedges, 2);

  // Vertex 3: should have 2 edges
  int v3_nedges = a2ab_h[4] - a2ab_h[3];
  TEST_EQUALITY_CONST(v3_nedges, 2);

  // Vertex 4: should have 1 edge
  int v4_nedges = a2ab_h[5] - a2ab_h[4];
  TEST_EQUALITY_CONST(v4_nedges, 1);
  TEST_EQUALITY_CONST(ab2b_h[a2ab_h[4]], 3); // edge 3
}

TEUCHOS_UNIT_TEST(OmegahGhost, getUpAdjacentEntsInClosureOfOwnedElms_1D_Parallel)
{
  // Create 1D mesh with 8 edges
  const int n_edges_total = 8;
  auto mesh = create1DMesh(n_edges_total);

  // Get vertex-to-edge upward adjacencies
  auto vtxToEdge = OmegahGhost::getUpAdjacentEntsInClosureOfOwnedElms(mesh, 0);
  auto vertsInClosure = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, 0);

  auto a2ab_h = Omega_h::HostRead(vtxToEdge.a2ab);
  auto ab2b_h = Omega_h::HostRead(vtxToEdge.ab2b);

  auto comm = Albany::getDefaultComm();
  auto numRanks = comm->getSize();
  auto rank = comm->getRank();
  const auto numVerts = a2ab_h.size()-1;
  if(numRanks > 1) {
    TEST_ASSERT(numRanks == 4); //hardcoded for four ranks
    if(rank==0) {
      const auto expectedNumAdjElms = std::array<int,4>({1,2,1,0});
      TEST_EQUALITY_CONST(numVerts, 4); //includes one ghost vtx
      for (int v = 0; v < numVerts; ++v) {
        int numAdj = a2ab_h[v + 1] - a2ab_h[v];
        TEST_EQUALITY_CONST(numAdj, expectedNumAdjElms[v]);
      }
      const auto expectedAdjElms = std::array<int,4>({0, 0,1, 1});
      TEST_EQUALITY_CONST(ab2b_h.size(), expectedAdjElms.size());
      for (int adjIdx = 0; adjIdx < ab2b_h.size(); ++adjIdx) {
        TEST_EQUALITY_CONST(ab2b_h[adjIdx], expectedAdjElms[adjIdx]);
      }
    } else if(rank == 1 || rank == 2) {
      TEST_EQUALITY_CONST(numVerts, 5); //includes two ghost verts
      const auto expectedNumAdjElms = std::array<int,5>({0,1,2,1,0});
      for (int v = 0; v < numVerts; ++v) {
        int numAdj = a2ab_h[v + 1] - a2ab_h[v];
        TEST_EQUALITY_CONST(numAdj, expectedNumAdjElms[v]);
      }
      const auto expectedAdjElms = std::array<int,4>({1, 1,2, 2});
      TEST_EQUALITY_CONST(ab2b_h.size(), expectedAdjElms.size());
      for (int adjIdx = 0; adjIdx < ab2b_h.size(); ++adjIdx) {
        TEST_EQUALITY_CONST(ab2b_h[adjIdx], expectedAdjElms[adjIdx]);
      }
    } else { //rank == 3
      TEST_EQUALITY_CONST(numVerts, 4); //includes one ghost vtx
      const auto expectedNumAdjElms = std::array<int,4>({0,1,2,1});
      for (int v = 0; v < numVerts; ++v) {
        int numAdj = a2ab_h[v + 1] - a2ab_h[v];
        TEST_EQUALITY_CONST(numAdj, expectedNumAdjElms[v]);
      }
      const auto expectedAdjElms = std::array<int,4>({1, 1,2, 2});
      TEST_EQUALITY_CONST(ab2b_h.size(), expectedAdjElms.size());
      for (int adjIdx = 0; adjIdx < ab2b_h.size(); ++adjIdx) {
        TEST_EQUALITY_CONST(ab2b_h[adjIdx], expectedAdjElms[adjIdx]);
      }
    }
  } else {
    return; //skip the test
  }

}

//
// TEST 10: Owned entities in closure
//
TEUCHOS_UNIT_TEST(OmegahGhost, getOwnedEntsInClosureOfOwnedElms)
{
  auto mesh = createBoxMesh2D(3, 3);

  // For elements, "owned in closure" should match owned mask
  auto ownedElmsInClosure = OmegahGhost::getOwnedEntsInClosureOfOwnedElms(mesh, mesh.dim());
  auto numOwnedElms = OmegahGhost::getNumOwnedElms(mesh);
  TEST_EQUALITY_CONST(ownedElmsInClosure.size(), numOwnedElms);

  // All should be marked as owned (value = 1)
  auto sum = Omega_h::get_sum(ownedElmsInClosure);
  TEST_EQUALITY_CONST(sum, numOwnedElms);

  // For vertices, check that owned vertices in closure <= total vertices in closure
  auto ownedVtxInClosure = OmegahGhost::getOwnedEntsInClosureOfOwnedElms(mesh, 0);
  auto numVtxInClosure = OmegahGhost::getNumEntsInClosureOfOwnedElms(mesh, 0);
  TEST_EQUALITY_CONST(ownedVtxInClosure.size(), numVtxInClosure);

  auto numOwnedVtxInClosure = Omega_h::get_sum(ownedVtxInClosure);
  TEST_ASSERT(numOwnedVtxInClosure <= numVtxInClosure);
}

//
// TEST 11: Element permutation from non-ghosted to ghosted indexing
//
TEUCHOS_UNIT_TEST(OmegahGhost, getElemPermutationFromNonGhostedToGhosted)
{
  // Create 2x2 2D triangular box mesh
  // need ghost elements interspersed with owned elements for a meaningful test
  auto cutAlongAxis = false;
  auto mesh = createBoxMesh2D(2, 2, cutAlongAxis);
  Omega_h::vtk::write_parallel("ghostBox2x2.osh", &mesh);

  auto comm = Albany::getDefaultComm();
  auto numRanks = comm->getSize();
  auto rank = comm->getRank();

  // Get the permutation array
  auto perm = OmegahGhost::getElemPermutationFromNonGhostedToGhosted(mesh);

  // Get number of owned elements
  auto numOwnedElms = OmegahGhost::getNumOwnedElms(mesh);

  // Permutation array should be sized for owned elements only
  TEST_EQUALITY_CONST(perm.size(), numOwnedElms);

  // Get owned element mask for validation
  auto isElmOwned = mesh.owned(mesh.dim());
  auto isElmOwned_h = Omega_h::HostRead(isElmOwned);
  auto perm_h = Omega_h::HostRead(perm);

  // Validate permutation properties:
  // 1. All values should be valid element indices
  for (int i = 0; i < perm_h.size(); ++i) {
    TEST_ASSERT(perm_h[i] >= 0);
    TEST_ASSERT(perm_h[i] < mesh.nelems());
  }

  // 2. All values should correspond to owned elements
  for (int i = 0; i < perm_h.size(); ++i) {
    TEST_EQUALITY_CONST(isElmOwned_h[perm_h[i]], 1);
  }

  // 3. All values should be unique (check by verifying each owned element appears exactly once)
  std::vector<int> counts(mesh.nelems(), 0);
  for (int i = 0; i < perm_h.size(); ++i) {
    counts[perm_h[i]]++;
  }
  for (int i = 0; i < perm_h.size(); ++i) {
    TEST_EQUALITY_CONST(counts[perm_h[i]], 1);
  }

  // Test specific values for 2 ranks
  if (numRanks == 4) {
    // check local element index permutation arrays for each rank
    if (rank == 0) {
      const std::vector<int> expected = {1, 4, 5};
      TEST_ASSERT(perm_h.size() == static_cast<int>(expected.size()));
      for (int i = 0; i < perm_h.size(); ++i) {
        TEST_EQUALITY_CONST(perm_h[i], expected[i]);
      }
    } else if (rank == 1) {
      TEST_ASSERT(perm_h.size() == 1);
      TEST_EQUALITY_CONST(perm_h[0], 2);
    } else if (rank == 2) {
      TEST_ASSERT(perm_h.size() == 1);
      TEST_EQUALITY_CONST(perm_h[0], 0);
    } else if (rank == 3) {
      const std::vector<int> expected = {3, 5, 6};
      TEST_ASSERT(perm_h.size() == static_cast<int>(expected.size()));
      for (int i = 0; i < perm_h.size(); ++i) {
        TEST_EQUALITY_CONST(perm_h[i], expected[i]);
      }
    }
  }
}
