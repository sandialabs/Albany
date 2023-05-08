//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

// I find assert-style checks more intuitive in unit tests.
// So throw if the input condition is false:
#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

// Check vectors are equal up to permutations
template<typename Vec1,
         typename Vec2>
bool sameAs(const Vec1& lhs,
            const Vec2& rhs)
{
  if ((long int)lhs.size()!=(long int)rhs.size()) return false;

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

    // Get stuff from disc
    const auto& sol_dof_mgr   = disc->getDOFManager();
    const auto& cell_indexer  = sol_dof_mgr->cell_indexer();

    const auto& nodeSets     = disc->getNodeSets();
    const auto& nodeSetsGIDs = disc->getNodeSetGIDs();

    const auto  stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc);
    const auto& bulk = stk_disc->getSTKBulkData();

    // Expected nodesets
    std::list<std::string> expected_nsn = {
      "NodeSet0", "NodeSet1", "NodeSet2", "NodeSet3", "NodeSet99"
    };
    std::map<std::string,std::vector<GO>> expected_ns_gids = {
      {"NodeSet0" ,  std::vector<GO>(N)},  // Left
      {"NodeSet1" ,  std::vector<GO>(N)},  // Right
      {"NodeSet2" ,  std::vector<GO>(N)},  // Bottom
      {"NodeSet3" ,  std::vector<GO>(N)},  // Top
      {"NodeSet99",  std::vector<GO>(1)}   // Bottom left corner
    };
    for (int i=0; i<N; ++i) {
      expected_ns_gids["NodeSet0"][i]=i*N;
      expected_ns_gids["NodeSet1"][i]=i*N+E;
      expected_ns_gids["NodeSet2"][i]=i;
      expected_ns_gids["NodeSet3"][i]=E*N+i;
    }
    expected_ns_gids["NodeSet99"] = {0};

    // Check nodesets
    constexpr auto NODE_RANK = stk::topology::NODE_RANK;
    REQUIRE (nodeSets.size()==5);
    REQUIRE (nodeSetsGIDs.size()==5);
    for (const auto& nsn : expected_nsn) {
      REQUIRE (nodeSets.find(nsn)!=nodeSets.end());
      REQUIRE (nodeSetsGIDs.find(nsn)!=nodeSetsGIDs.end());

      const auto& ns_node_elem_pos  = nodeSets.at(nsn);
      const auto& ns_nodes = nodeSetsGIDs.at(nsn);

      TEST_EQUALITY (ns_nodes.size(),ns_node_elem_pos.size());
      const int num_local_nodes = ns_nodes.size();
      int num_global_nodes;
      Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&num_local_nodes,&num_global_nodes);
      
      REQUIRE (num_global_nodes==static_cast<int>(expected_ns_gids.at(nsn).size()));

      std::vector<GO> global_ns_nodes(num_global_nodes);
      Albany::all_gather_v(ns_nodes.data(),num_local_nodes,global_ns_nodes.data(),comm);
      REQUIRE (sameAs(global_ns_nodes,expected_ns_gids[nsn]));

      for (int i=0; i<num_local_nodes; ++i) {
        const auto& n = bulk.get_entity(NODE_RANK,ns_nodes[i]+1);
        const auto e = *bulk.begin_elements(n);
        const auto pos = stk_disc->determine_entity_pos(e,n);
        const auto elem_lid = cell_indexer->getLocalElement(stk_disc->stk_gid(e));
        REQUIRE (ns_node_elem_pos[i].first==elem_lid);
        REQUIRE (ns_node_elem_pos[i].second==pos);
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

TEUCHOS_UNIT_TEST(STKDiscTests, JacGraph)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto comm = Albany::getDefaultComm();

  // Simple cube-like geometry with E elements and N=E+1 nodes per dimension.
  // In 2D, the node numbering would look like the following
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
  const int E = 2;
  const int N = E+1;
  const int num_dims = 3;

  auto run = [&] (int neq) {
    // Create disc
    const auto disc = UnitTest::createTestDisc (comm, num_dims, E, neq);
    const auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc,true);
    const auto sol_dof_mgr = disc->getDOFManager();
    const auto ov_sol_indexer = sol_dof_mgr->ov_indexer();
    const auto cell_indexer = sol_dof_mgr->cell_indexer();

    // Create jacobian
    const auto J = disc->createJacobianOp();

    // Create expected graph pattern by looping over elems,
    // and adding a dense pattern to all rows of gids in that element
    std::vector<std::vector<GO>> expected_gids(std::pow(N,num_dims)*neq);
    std::vector<std::vector<int>> expected_lids(std::pow(N,num_dims)*neq);

    for (int elem_GID=0; elem_GID<std::pow(E,num_dims); ++elem_GID) {
      const LO elem_LID = cell_indexer->getLocalElement(elem_GID);
      if (elem_LID<0) continue;

      const auto& elem_dof_gids = sol_dof_mgr->getElementGIDs(elem_LID);
      for (auto row : elem_dof_gids) {
        for (auto col : elem_dof_gids) {
          expected_gids[row].push_back(col);
        }
      }
    }

    // Perform manual global assembly:
    std::vector<GO> my_rows;
    for (GO row=0; row<std::pow(N,num_dims)*neq; ++row) {
      auto& my_row = expected_gids[row];
      const int row_lid = sol_dof_mgr->indexer()->getLocalElement(row);
      std::set<GO> glob_row;
      for (int pid=0; pid<comm->getSize(); ++pid) {
        int ncols = my_row.size();
        Teuchos::broadcast(*comm,pid,1,&ncols);
        std::vector<GO> cols (ncols);
        if (pid==comm->getRank()) {
          cols = my_row;
        }

        Teuchos::broadcast(*comm,pid,ncols,cols.data());
        for (GO col : cols) {
          glob_row.insert(col);
        }
      }
      if (row_lid>=0) {
        my_rows.push_back(row);
        my_row.clear();
        for (GO col : glob_row) {
          my_row.push_back(col);
        }
      }
    }

    // Remove duplicates, and convert to lids
    const auto Tmat = Albany::getTpetraMatrix(J,true);
    const auto colMap = Tmat->getColMap();
    for (int i=0; i<neq*std::pow(N,num_dims); ++i) {
      auto& v = expected_gids[i];
      std::sort(v.begin(),v.end());
      auto it = std::unique(v.begin(),v.end());
      v.erase(it,v.end());
      for (auto gid : v) {
        expected_lids[i].push_back(colMap->getLocalElement(gid));
      }
    }

    // Now check that the pattern of J matches the one we built by hand
    auto range = J->range();
    const int num_local_rows = Albany::getLocalSubdim(range);
    Teuchos::Array<LO> col_lids;
    Teuchos::Array<ST> vals;
    for (int irow=0; irow<num_local_rows; ++irow) {
      const GO row_gid = ov_sol_indexer->getGlobalElement(irow);
      Albany::getLocalRowValues (J,irow,col_lids,vals);

      REQUIRE (sameAs(expected_lids[row_gid],col_lids));
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
