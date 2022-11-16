//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

// I find assert-style checks more intuitive in unit tests.
// So throw if the input condition is false:
#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

// Check vectors are equal up to permutations
template<typename T>
bool sameAs(const std::vector<T>& lhs,
            const std::vector<T>& rhs)
{
  if (lhs.size()!=rhs.size()) return false;

  for (auto l : lhs) {
    auto it = std::find(rhs.begin(),rhs.end(),l);
    if (it==rhs.end()) {
      return false;
    }
  }

  return true;
}

TEUCHOS_UNIT_TEST(STKDiscTests, NodeSets)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto comm = Albany::getDefaultComm();

  // Simple square 2d geometry with E elements and N=E+1 nodes per side.
  //
  //     E*N ----- E*N+1 ----- E*N+2 .....  N^2-1
  //      .          .           .            .
  //      .          .           .            .
  //      .          .           .            .
  //     2N ------ 2N+1 ------ 2N+2 .......  E+2N
  //      |          |           |            |
  //      N ------  N+1 ------  N+2 .......  E+N
  //      |          |           |            |
  //      0 ------   1  ------   2  .......   E
  //

  // Some constants
  const int E = 2*comm->getSize();  // #elems per side
  const int N = E+1;                // #nodes per side
  const int num_dims = 2;

  auto run = [&](const int neq) {

    // Create disc
    const auto  disc = UnitTest::createTestDisc (comm, num_dims, E, neq);
    const auto& sol_name     = disc->solution_dof_name();
    const auto& sol_indexer  = disc->getNewDOFManager()->indexer();
    const auto& nodeSets     = disc->getNodeSets();
    const auto& nodeSetsGIDs = disc->getNodeSetGIDs();

    // Expected nodesets
    std::list<std::string> expected_nsn = {
      "NodeSet0", "NodeSet1", "NodeSet2", "NodeSet3", "NodeSet99"
    };
    std::map<std::string,std::vector<GO>> expected_ns_gids = {
      {"NodeSet0" ,  std::vector<GO>(N)},  // Left
      {"NodeSet1" ,  std::vector<GO>(N)},  // Right
      {"NodeSet2" ,  std::vector<GO>(N)},  // Bottom
      {"NodeSet3" ,  std::vector<GO>(N)},   // Top
      {"NodeSet99",  std::vector<GO>(4)}   // Corners
    };
    for (int i=0; i<N; ++i) {
      expected_ns_gids["NodeSet0"][i]=i*N;
      expected_ns_gids["NodeSet1"][i]=i*N+E;
      expected_ns_gids["NodeSet2"][i]=i;
      expected_ns_gids["NodeSet3"][i]=E*N+i;
    }
    expected_ns_gids["NodeSet99"] = {0};

    // Check nodesets
    REQUIRE (nodeSets.size()==5);
    REQUIRE (nodeSetsGIDs.size()==5);
    std::vector<GO> ns_dofs_gids;
    for (const auto& nsn : expected_nsn) {
      REQUIRE (nodeSets.find(nsn)!=nodeSets.end());
      REQUIRE (nodeSetsGIDs.find(nsn)!=nodeSetsGIDs.end());

      const auto& ns_dof_mgr   = disc->getNewDOFManager(sol_name,nsn);

      const auto& ns_dofs  = nodeSets.at(nsn);
      const auto& ns_nodes = nodeSetsGIDs.at(nsn);

      TEST_EQUALITY (ns_nodes.size(),ns_dofs.size());
      const int num_local_nodes = ns_nodes.size();
      int num_global_nodes;
      Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&num_local_nodes,&num_global_nodes);

      REQUIRE (num_global_nodes==static_cast<int>(expected_ns_gids.at(nsn).size()));

      auto dof_mgr_elems = ns_dof_mgr->getAlbanyConnManager()->getElementsInBlock(nsn);
      REQUIRE (sameAs(dof_mgr_elems,ns_nodes));

      for (int i=0; i<num_local_nodes; ++i) {

        const auto node_gid = ns_nodes[i];
        const auto it = std::find(expected_ns_gids[nsn].begin(),expected_ns_gids[nsn].end(),node_gid);
        REQUIRE(it!=expected_ns_gids[nsn].end());

        const auto node_lid = ns_dof_mgr->cell_indexer()->getLocalElement(node_gid);
        ns_dof_mgr->getElementGIDs(node_lid,ns_dofs_gids);

        REQUIRE (static_cast<int>(ns_dofs[i].size())==neq);
        REQUIRE (static_cast<int>(ns_dofs_gids.size())==neq);
        for (int eq=0; eq<neq; ++eq) {
          REQUIRE (sol_indexer->getLocalElement(ns_dofs_gids[eq])==ns_dofs[node_lid][eq]);
        }
      }
    }
  };

  // Single equation case
  run (1);

  // Multiple equations case
  run (3);

  // Silence compiler warnings due to unused stuff from Teuchos testing framework.
  (void) out;
  (void) success;
}

TEUCHOS_UNIT_TEST(STKDiscTests, JacPattern)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto comm = Albany::getDefaultComm();

  // Simple square 2d geometry with E elements and N=E+1 nodes per side.
  //
  //     EN ------ EN+1 ------ EN+2 ....... EN+E
  //      .          .           .            .
  //      .          .           .            .
  //      .          .           .            .
  //     2N ------ 2N+1 ------ 2N+2 ....... 2N+E
  //      |          |           |            |
  //      N ------  N+1 ------  N+2 .......  N+E
  //      |          |           |            |
  //      0 ------   1  ------   2  .......   E
  //

  // Some constants
  const int E = 2*comm->getSize();
  const int N = E+1;
  const int num_dims = 2;

  auto run = [&] (int neq) {
    // Create disc
    auto disc = UnitTest::createTestDisc (comm, num_dims, E, neq);
    auto node_indexer = disc->getNodeNewDOFManager()->indexer();
    auto ov_sol_indexer = disc->getNewDOFManager()->ov_indexer();
    int num_my_nodes = Albany::getLocalSubdim(node_indexer->getVectorSpace());

    // Create jacobian
    auto J = disc->createJacobianOp();

    // Create expected graph pattern. We expect
    //  - 4*neq nonzeros on rows at corners
    //  - 6*neq nonzeros on rows at edges
    //  - 9*neq nonzeros on rows at internal nodes
    std::vector<int> expected_nnz(num_my_nodes*neq);
    std::vector<std::vector<int>> expected_lids(num_my_nodes*neq);
    auto lid = [&](const GO gid) -> LO {
      return ov_sol_indexer->getLocalElement(gid);
    };

    auto add_lids = [&](std::vector<int>& lids, const GO node_gid) {
      for (int eq=0; eq<neq; ++eq) {
        lids.push_back(lid(node_gid*neq+eq));
      }
    };
    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        GO node_gid = i*N+j;
        LO node_lid = node_indexer->getLocalElement(node_gid);
        if (node_lid<0) {
          continue;
        }

        // The easy task: compute nnz
        if (i % N == 0 || i % N == E) {
          if (j % N == 0 || j % N == E) {
            // Corner
            expected_nnz[node_lid] = neq*4;
          } else {
            // Horiz Edge
            expected_nnz[node_lid] = neq*6;
          }
        } else if (j % N == 0 || j % N == E) {
          // Vert edge
          expected_nnz[node_lid] = neq*6;
        } else {
          // Internal
          expected_nnz[node_lid] = neq*9;
        }

        // The boring task: get actual lids
        for (int eq=0; eq<neq; ++eq) {
          const LO dof_lid = ov_sol_indexer->getLocalElement(node_gid*neq+eq);
          auto& lids = expected_lids[dof_lid];
          if (i%N == 0) {
            if (j%N == 0) {
              // Bot left
              add_lids(lids,0);
              add_lids(lids,1);
              add_lids(lids,E+1);
              add_lids(lids,E+2);
            } else if (j%N == E) {
              // Top left
              add_lids(lids,E*(E+1));
              add_lids(lids,E*(E+1)+1);
              add_lids(lids,(E-1)*(E+1));
              add_lids(lids,(E-1)*(E+1)+1);
            } else {
              // Left
              add_lids(lids,j*(E+1));
              add_lids(lids,(j-1)*(E+1));
              add_lids(lids,(j+1)*(E+1));
              add_lids(lids,j*(E+1)+1);
              add_lids(lids,(j-1)*(E+1)+1);
              add_lids(lids,(j+1)*(E+1)+1);
            }
          } else if (i%N == E) {
            if (j%N == 0) {
              // Bot right
              add_lids(lids,E);
              add_lids(lids,E+1);
              add_lids(lids,2*E);
              add_lids(lids,2*E+1);
            } else if (j%N == E) {
              // Top right
              add_lids(lids,E*(E+1)-2);
              add_lids(lids,E*(E+1)-1);
              add_lids(lids,(E+1)*(E+1)-2);
              add_lids(lids,(E+1)*(E+1)-1);
            } else {
              // Right
              add_lids(lids,(j-1)*(E+1)+E);
              add_lids(lids,j*(E+1)+E);
              add_lids(lids,(j+1)*(E+1)+E);
              add_lids(lids,(j-1)*(E+1)+E-1);
              add_lids(lids,j*(E+1)+E-1);
              add_lids(lids,(j+1)*(E+1)+E-1);
            }
          } else {
            if (j%N == 0) {
              // Bottom
              add_lids(lids,i-1);
              add_lids(lids,i);
              add_lids(lids,i+1);
              add_lids(lids,E+1+i-1);
              add_lids(lids,E+1+i);
              add_lids(lids,E+1+i+1);
            } else if (j%N == E) {
              // Top
              add_lids(lids,E*(E+1)+i-1);
              add_lids(lids,E*(E+1)+i);
              add_lids(lids,E*(E+1)+i+1);
              add_lids(lids,(E-1)*(E+1)+i-1);
              add_lids(lids,(E-1)*(E+1)+i);
              add_lids(lids,(E-1)*(E+1)+i+1);
            } else {
              // Internal
              add_lids(lids,(j-1)*(E+1)+i-1);
              add_lids(lids,(j-1)*(E+1)+i);
              add_lids(lids,(j-1)*(E+1)+i+1);
              add_lids(lids,j*(E+1)+i-1);
              add_lids(lids,j*(E+1)+i);
              add_lids(lids,j*(E+1)+i+1);
              add_lids(lids,(j+1)*(E+1)+i-1);
              add_lids(lids,(j+1)*(E+1)+i);
              add_lids(lids,(j+1)*(E+1)+i+1);
            }
          }
        }
      }
    }

    auto range = J->range();
    const int num_local_rows = Albany::getLocalSubdim(range);
    Teuchos::Array<LO> col_lids;
    Teuchos::Array<ST> vals;
    for (int irow=0; irow<num_local_rows; ++irow) {
      Albany::getLocalRowValues (J,irow,col_lids,vals);
      REQUIRE(col_lids.size()==expected_nnz[irow]);

      std::cout << "comparing:\n";
      std::cout << "  expected:";
      for (auto v : expected_lids[irow]) {
        std::cout << " " << v;
      }
      std::cout << "\n  actual:";
      for (auto v : col_lids) {
        std::cout << " " << v;
      }
      std::cout << "\n";
      std::sort(expected_lids[irow].begin(),expected_lids[irow].end());
      std::sort(col_lids.begin(),col_lids.end());

      for (int inz=0; inz<expected_nnz[irow]; ++inz) {
        REQUIRE(col_lids[inz]==expected_lids[irow][inz]);
      }
    }
  };

  // Run single eq case
  run(1);

  // Run multiple eq case
  run(3);

  // Silence compiler warnings due to unused stuff from Teuchos testing framework.
  (void) out;
  (void) success;
}
