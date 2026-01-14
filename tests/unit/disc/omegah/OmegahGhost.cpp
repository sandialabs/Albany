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

// Helper function to create 2D simplex box mesh
Omega_h::Mesh createBoxMesh2D(int nx, int ny) {
  auto lib = Albany::get_omegah_lib();
  const auto scale = 1.0;
  const auto nz = 0;
  auto mesh = Omega_h::build_box(lib.world(),OMEGA_H_SIMPLEX,
                                          scale,scale,scale,
                                          nx,ny,nz);
  const auto verbose = true;
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
  const auto verbose = true;
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
TEUCHOS_UNIT_TEST(OmegahGhost, getOwnedEntityGids_Serial)
{
  auto mesh = createBoxMesh2D(3, 3);

  // Test vertices (dim = 0)
  auto vtxGids = OmegahGhost::getOwnedEntityGids(mesh, 0);
  int expectedVtx = 16; // 4x4 grid of vertices
  TEST_EQUALITY_CONST(vtxGids.size(), expectedVtx);

  // Test edges (dim = 1)
  auto edgeGids = OmegahGhost::getOwnedEntityGids(mesh, 1);
  int expectedEdges = 33; // 3*(3+1) + 4*3 = 33 for 3x3 tri mesh
  TEST_EQUALITY_CONST(edgeGids.size(), expectedEdges);

  // Test triangles (dim = 2)
  auto triGids = OmegahGhost::getOwnedEntityGids(mesh, 2);
  int expectedTris = 18; // 3*3*2 = 18 triangles
  TEST_EQUALITY_CONST(triGids.size(), expectedTris);

  // Verify GIDs are unique (check for duplicates)
  std::set<Omega_h::GO> uniqueGids(vtxGids.begin(), vtxGids.end());
  TEST_EQUALITY_CONST((int)uniqueGids.size(), expectedVtx);
}

TEUCHOS_UNIT_TEST(OmegahGhost, getOwnedEntityGids_Parallel)
{
  auto comm = Albany::getDefaultComm();
  auto mesh = createBoxMesh2D(3, 3);

  // Test vertices
  auto vtxGids = OmegahGhost::getOwnedEntityGids(mesh, 0);

  // Each rank should own some vertices
  TEST_ASSERT(vtxGids.size() > 0);

  // Gather all GIDs to check uniqueness
  int localCount = vtxGids.size();
  int globalCount;
  Teuchos::reduceAll<int>(*comm, Teuchos::REDUCE_SUM, 1, &localCount, &globalCount);

  // Should equal total vertices
  TEST_EQUALITY_CONST(globalCount, 16);
}

//
// TEST 3: Entities in closure - serial
//
TEUCHOS_UNIT_TEST(OmegahGhost, getEntsInClosureOfOwnedElms_Serial)
{
  auto mesh = createBoxMesh2D(2, 2);

  // In serial with no ghosts, ALL entities are in closure of owned elements

  // Test vertices
  auto vtxMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, 0);
  TEST_EQUALITY_CONST(vtxMask.size(), mesh.nverts());
  auto vtxSum = Omega_h::get_sum(vtxMask);
  TEST_EQUALITY_CONST(vtxSum, mesh.nverts());

  // Test edges
  auto edgeMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, 1);
  TEST_EQUALITY_CONST(edgeMask.size(), mesh.nedges());
  auto edgeSum = Omega_h::get_sum(edgeMask);
  TEST_EQUALITY_CONST(edgeSum, mesh.nedges());

  // Test triangles
  auto triMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, 2);
  TEST_EQUALITY_CONST(triMask.size(), mesh.nelems());
  auto triSum = Omega_h::get_sum(triMask);
  TEST_EQUALITY_CONST(triSum, mesh.nelems());
}

//
// TEST 4: Entities in closure - parallel (key ghost test)
//
TEUCHOS_UNIT_TEST(OmegahGhost, getEntsInClosureOfOwnedElms_Parallel)
{
  auto mesh = createBoxMesh2D(4, 4);

  // With ghosts, some entities may NOT be in closure of owned elements

  // Test elements
  auto elmMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, mesh.dim());
  auto numInClosure = Omega_h::get_sum(elmMask);
  auto numOwned = OmegahGhost::getNumOwnedElms(mesh);

  // For elements, "in closure" should equal owned
  TEST_EQUALITY_CONST(numInClosure, numOwned);

  // Test vertices
  auto vtxMask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, 0);
  auto numVtxInClosure = Omega_h::get_sum(vtxMask);
  auto totalVtx = mesh.nverts();

  // Should have some vertices in closure
  TEST_ASSERT(numVtxInClosure > 0);
  TEST_ASSERT(numVtxInClosure <= totalVtx);
}

//
// TEST 5: Count entities in closure
//
TEUCHOS_UNIT_TEST(OmegahGhost, getNumEntsInClosureOfOwnedElms)
{
  auto mesh = createBoxMesh2D(3, 3);

  // Test that count matches mask sum
  for (int dim = 0; dim <= mesh.dim(); ++dim) {
    auto mask = OmegahGhost::getEntsInClosureOfOwnedElms(mesh, dim);
    auto maskSum = Omega_h::get_sum(mask);
    auto count = OmegahGhost::getNumEntsInClosureOfOwnedElms(mesh, dim);

    TEST_EQUALITY_CONST(count, maskSum);
  }
}

//
// TEST 6: GIDs in closure
//
TEUCHOS_UNIT_TEST(OmegahGhost, getEntGidsInClosureOfOwnedElms)
{
  auto mesh = createBoxMesh2D(2, 2);

  for (int dim = 0; dim <= mesh.dim(); ++dim) {
    auto gids = OmegahGhost::getEntGidsInClosureOfOwnedElms(mesh, dim);
    auto count = OmegahGhost::getNumEntsInClosureOfOwnedElms(mesh, dim);

    // Length should match count
    TEST_EQUALITY_CONST(gids.size(), count);

    // All GIDs should be valid (non-negative)
    for (int i = 0; i < gids.size(); ++i) {
      TEST_ASSERT(gids[i] >= 0);
    }
  }
}

//
// TEST 7: Vertex coordinates in closure
//
TEUCHOS_UNIT_TEST(OmegahGhost, getVtxCoordsInClosureOfOwnedElms)
{
  auto mesh = createBoxMesh2D(2, 2);

  auto coords = OmegahGhost::getVtxCoordsInClosureOfOwnedElms(mesh);
  auto numVtxInClosure = OmegahGhost::getNumEntsInClosureOfOwnedElms(mesh, 0);

  // Should have dim * numVertices entries
  TEST_EQUALITY_CONST(coords.size(), numVtxInClosure * mesh.dim());

  // Check coordinates are within box bounds [0,1] x [0,1]
  auto coords_h = Omega_h::HostRead(coords);
  for (int i = 0; i < coords_h.size(); ++i) {
    TEST_ASSERT(coords_h[i] >= 0.0);
    TEST_ASSERT(coords_h[i] <= 1.0);
  }
}

//
// TEST 8: Downward Adjacency (Edge→Vertex) for 1D Mesh
//
TEUCHOS_UNIT_TEST(OmegahGhost, getDownAdjacentEntsInClosureOfOwnedElms_1D_Serial)
{
  // Create 1D mesh with 4 edges: 0--1--2--3--4
  const int n_edges = 4;
  auto mesh = create1DMesh(n_edges);

  // Get edge-to-vertex downward adjacencies for owned edges
  // dim=0 means we're getting vertices (downward from edges)
  auto edgeToVtx = OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(mesh, 0);
  auto numOwnedEdges = OmegahGhost::getNumOwnedElms(mesh);

  // Should have 2 vertices per edge
  TEST_EQUALITY_CONST(edgeToVtx.size(), numOwnedEdges * 2);
  TEST_EQUALITY_CONST(numOwnedEdges, n_edges);

  // Check connectivity is correct
  auto edgeToVtx_h = Omega_h::HostRead(edgeToVtx);
  for (int e = 0; e < n_edges; ++e) {
    int v0 = edgeToVtx_h[e * 2 + 0];
    int v1 = edgeToVtx_h[e * 2 + 1];

    // Edge e should connect vertices e and e+1
    TEST_EQUALITY_CONST(v0, e);
    TEST_EQUALITY_CONST(v1, e + 1);
  }
}

TEUCHOS_UNIT_TEST(OmegahGhost, getDownAdjacentEntsInClosureOfOwnedElms_1D_Parallel)
{
  auto comm = Albany::getDefaultComm();

  // Create 1D mesh with 8 edges total
  const int n_edges_total = 8;
  auto mesh = create1DMesh(n_edges_total);

  // Get edge-to-vertex downward adjacencies for owned edges only
  auto edgeToVtx = OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(mesh, 0);
  auto numOwnedEdges = OmegahGhost::getNumOwnedElms(mesh);

  // Should have 2 vertices per owned edge
  TEST_EQUALITY_CONST(edgeToVtx.size(), numOwnedEdges * 2);

  // Each rank should own some edges
  TEST_ASSERT(numOwnedEdges > 0);

  // Check all vertex indices are valid
  auto edgeToVtx_h = Omega_h::HostRead(edgeToVtx);
  int nverts = mesh.nverts();
  for (int i = 0; i < edgeToVtx_h.size(); ++i) {
    TEST_ASSERT(edgeToVtx_h[i] >= 0);
    TEST_ASSERT(edgeToVtx_h[i] < nverts);
  }

  // Sum across ranks should equal total edges
  int globalOwnedEdges;
  Teuchos::reduceAll<int>(*comm, Teuchos::REDUCE_SUM, 1, &numOwnedEdges, &globalOwnedEdges);
  TEST_EQUALITY_CONST(globalOwnedEdges, n_edges_total);
}

//
// TEST 9: Upward Adjacency (Vertex→Edge) for 1D Mesh
//
TEUCHOS_UNIT_TEST(OmegahGhost, getUpAdjacentEntsInClosureOfOwnedElms_1D_Serial)
{
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
  auto numVtxInClosure = OmegahGhost::getNumEntsInClosureOfOwnedElms(mesh, 0);

  // Graph should have entries for vertices in closure
  TEST_EQUALITY_CONST(vtxToEdge.a2ab.size(), numVtxInClosure + 1);

  auto a2ab_h = Omega_h::HostRead(vtxToEdge.a2ab);
  auto ab2b_h = Omega_h::HostRead(vtxToEdge.ab2b);

  // Verify graph structure consistency:
  // - All adjacency counts should be reasonable (1 or 2 for 1D line mesh)
  // - All edge indices should be valid

  for (int v = 0; v < numVtxInClosure; ++v) {
    int numAdj = a2ab_h[v + 1] - a2ab_h[v];

    // In a 1D line mesh, vertices should have 1 or 2 adjacent edges
    TEST_ASSERT(numAdj >= 1);
    TEST_ASSERT(numAdj <= 2);

    // Check all edge indices are valid
    for (int j = a2ab_h[v]; j < a2ab_h[v + 1]; ++j) {
      int edge = ab2b_h[j];
      TEST_ASSERT(edge >= 0);
      // Note: edge index is in the filtered space (owned edges only)
      TEST_ASSERT(edge < OmegahGhost::getNumOwnedElms(mesh));
    }
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
