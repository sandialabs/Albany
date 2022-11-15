//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"

#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

TEUCHOS_UNIT_TEST(JacPattern, STKDiscTests)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto comm = Albany::getDefaultComm();

  // Simple 2d geometry, with the following nodes numbering
  //
  //  N*(N+1) -- N*(N+1)+1 --- N*(N+1)+2 ...(N+1)^2-1
  //      .           .          .             .
  //      .           .          .             .
  //      .           .          .             .
  //    2N+2 ------ 2N+3 ------ 2N+4 ....... 3N+2
  //      |           |           |            |
  //     N+1 ------  N+2 ------  N+3 ....... 2N+1
  //      |           |           |            |
  //      0  ------   1  ------   2  .......   N
  //

  // Some constants
  const int N = 2*comm->getSize();
  const int M = N+1;
  const int num_dims = 2;

  auto run = [&] (int neq) {
    // Create disc
    auto disc = UnitTest::createTestDisc (comm, num_dims, N, neq);
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
    for (int i=0; i<M; ++i) {
      for (int j=0; j<M; ++j) {
        GO node_gid = i*M+j;
        LO node_lid = node_indexer->getLocalElement(node_gid);
        if (node_lid<0) {
          continue;
        }

        // The easy task: compute nnz
        if (i % M == 0 || i % M == N) {
          if (j % M == 0 || j % M == N) {
            // Corner
            expected_nnz[node_lid] = neq*4;
          } else {
            // Horiz Edge
            expected_nnz[node_lid] = neq*6;
          }
        } else if (j % M == 0 || j % M == N) {
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
          if (i%M == 0) {
            if (j%M == 0) {
              // Bot left
              add_lids(lids,0);
              add_lids(lids,1);
              add_lids(lids,N+1);
              add_lids(lids,N+2);
            } else if (j%M == N) {
              // Top left
              add_lids(lids,N*(N+1));
              add_lids(lids,N*(N+1)+1);
              add_lids(lids,(N-1)*(N+1));
              add_lids(lids,(N-1)*(N+1)+1);
            } else {
              // Left
              add_lids(lids,j*(N+1));
              add_lids(lids,(j-1)*(N+1));
              add_lids(lids,(j+1)*(N+1));
              add_lids(lids,j*(N+1)+1);
              add_lids(lids,(j-1)*(N+1)+1);
              add_lids(lids,(j+1)*(N+1)+1);
            }
          } else if (i%M == N) {
            if (j%M == 0) {
              // Bot right
              add_lids(lids,N);
              add_lids(lids,N+1);
              add_lids(lids,2*N);
              add_lids(lids,2*N+1);
            } else if (j%M == N) {
              // Top right
              add_lids(lids,N*(N+1)-2);
              add_lids(lids,N*(N+1)-1);
              add_lids(lids,(N+1)*(N+1)-2);
              add_lids(lids,(N+1)*(N+1)-1);
            } else {
              // Right
              add_lids(lids,(j-1)*(N+1)+N);
              add_lids(lids,j*(N+1)+N);
              add_lids(lids,(j+1)*(N+1)+N);
              add_lids(lids,(j-1)*(N+1)+N-1);
              add_lids(lids,j*(N+1)+N-1);
              add_lids(lids,(j+1)*(N+1)+N-1);
            }
          } else {
            if (j%M == 0) {
              // Bottom
              add_lids(lids,i-1);
              add_lids(lids,i);
              add_lids(lids,i+1);
              add_lids(lids,N+1+i-1);
              add_lids(lids,N+1+i);
              add_lids(lids,N+1+i+1);
            } else if (j%M == N) {
              // Top
              add_lids(lids,N*(N+1)+i-1);
              add_lids(lids,N*(N+1)+i);
              add_lids(lids,N*(N+1)+i+1);
              add_lids(lids,(N-1)*(N+1)+i-1);
              add_lids(lids,(N-1)*(N+1)+i);
              add_lids(lids,(N-1)*(N+1)+i+1);
            } else {
              // Internal
              add_lids(lids,(j-1)*(N+1)+i-1);
              add_lids(lids,(j-1)*(N+1)+i);
              add_lids(lids,(j-1)*(N+1)+i+1);
              add_lids(lids,j*(N+1)+i-1);
              add_lids(lids,j*(N+1)+i);
              add_lids(lids,j*(N+1)+i+1);
              add_lids(lids,(j+1)*(N+1)+i-1);
              add_lids(lids,(j+1)*(N+1)+i);
              add_lids(lids,(j+1)*(N+1)+i+1);
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
      TEST_EQUALITY_CONST(col_lids.size(),expected_nnz[irow]);

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
        TEST_EQUALITY_CONST(col_lids[inz],expected_lids[irow][inz]);
      }
    }
  };

  // Run single eq case
  run(1);

  // Run multiple eq case
  run(3);

  // Silence compiler warnings due to unused stuff coming from Teuchos testing framework.
  (void) out;
  (void) success;
}
