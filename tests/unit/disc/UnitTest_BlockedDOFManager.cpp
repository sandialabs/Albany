//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <set>

#include "Panzer_BlockedDOFManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_PauseToAttach.hpp"

#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "STKConnManager.hpp"
#include "Albany_BlockedSTKDiscretization.hpp"

// include some intrepid basis functions
// 2D basis
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"

#include "Kokkos_DynRankView.hpp"

#include "Albany_ThyraUtils.hpp"

#include "Albany_IossSTKMeshStruct.hpp"

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;

typedef Kokkos::DynRankView<double, PHX::Device> FieldContainer;

namespace Albany
{

   template <typename Intrepid2Type>
   Teuchos::RCP<const panzer::FieldPattern> buildFieldPattern()
   {
      // build a geometric pattern from a single basis
      Teuchos::RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis = rcp(new Intrepid2Type);
      Teuchos::RCP<const panzer::FieldPattern> pattern = rcp(new panzer::Intrepid2FieldPattern(basis));
      return pattern;
   }

   TEUCHOS_UNIT_TEST(AlbanyBlockedDOFManager_SimpleTests, assortedTests)
   {

      /*

GAH Design note:

The exodus read and building of the ConnManager and DOFManager should be done as a function, or perhaps the first test.

These structures then should be tested in small, distinct unit test sections instead of this large test here. These
tests are a beginning, "work in progress."

*/

      // Set the static variable that denotes this as a Tpetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));

      Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();
      int numProcs = comm->getSize();
      int myRank = comm->getRank();

      // panzer::pauseToAttach();

      using Teuchos::RCP;
      using Teuchos::rcp;
      using Teuchos::rcp_dynamic_cast;

      // Build an STK Discretization object that holds the test mesh "2D_Blk_Test.e"

      // 2D, quad, 3 block mesh built in Cubit.
      Teuchos::RCP<Teuchos::ParameterList> discParams = rcp(new Teuchos::ParameterList);
      discParams->set<std::string>("Exodus Input File Name", "2D_Blk_Test.e");
      discParams->set<std::string>("Method", "Exodus");
      discParams->set<int>("Interleaved Ordering", 2);
      discParams->set<bool>("Use Composite Tet 10", 0);
      discParams->set<int>("Number Of Time Derivatives", 0);
      discParams->set<bool>("Use Serial Mesh", 1);

      // Need to test various meshes, with various element types and block structures.

      Teuchos::RCP<Albany::AbstractMeshStruct> meshStruct;
      meshStruct = DiscretizationFactory::createMeshStruct(discParams, comm, 0);
      auto ms = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct);
      auto gms = Teuchos::rcp_dynamic_cast<GenericSTKMeshStruct>(meshStruct);

      const AbstractFieldContainer::FieldContainerRequirements req;
      const Teuchos::RCP<StateInfoStruct> sis = Teuchos::rcp(new StateInfoStruct());
      const std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> side_set_sis;
      const std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> side_set_req;

      ms->setFieldAndBulkData(comm, discParams, 3, req, sis, AbstractMeshStruct::DEFAULT_WORKSET_SIZE,
                              side_set_sis, side_set_req);

      // Null for this test
      const Teuchos::RCP<Albany::RigidBodyModes> rigidBodyModes;
      const std::map<int, std::vector<std::string>> sideSetEquations;

      // Use the Albany STK interface as it is used elsewhere in the code
      auto stkDisc = Teuchos::rcp(new STKDiscretization(discParams, ms, comm, rigidBodyModes, sideSetEquations));
      stkDisc->updateMesh();

      // Connection manager is the interface between Albany's historical STK interface and the Panzer DOF manager (and the
      // BlockedSTKDiscretization interface

      RCP<Albany::STKConnManager> connManager = rcp(new Albany::STKConnManager(ms));
      panzer::BlockedDOFManager dofManager;
      dofManager.setUseDOFManagerFEI(false);
      // Fix to use Teuchos communicator interface
      dofManager.setConnManager(connManager, MPI_COMM_WORLD);

      TEST_ASSERT(dofManager.getComm() != Teuchos::null);
      TEST_EQUALITY(dofManager.getConnManager(), connManager);
      TEST_EQUALITY(dofManager.getMaxSubFieldNumber(), -1);

      RCP<const panzer::FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::exec_space, double, double>>();

      dofManager.addField("T", patternC1); // add it to all three blocks

      // These are the element block names set in the exodus file by Cubit (look at the jou file)
      dofManager.addField("left_lower_qtr", "Ux", patternC1);
      dofManager.addField("left_lower_qtr", "Uy", patternC1);
      dofManager.addField("left_lower_qtr", "P", patternC1);

      dofManager.addField("right_half", "rho", patternC1);

      // Build the connectivity

      std::cout << "1. Building the connectivity" << std::endl;
      connManager->buildConnectivity(*patternC1);

      std::vector<std::string> eBlocks;
      dofManager.getElementBlockIds(eBlocks);
      std::sort(eBlocks.begin(), eBlocks.end());
      TEST_EQUALITY(eBlocks.size(), 3);
      TEST_EQUALITY(eBlocks[0], "left_lower_qtr");
      TEST_EQUALITY(eBlocks[1], "left_upper_qtr");
      TEST_EQUALITY(eBlocks[2], "right_half");
      TEST_EQUALITY(dofManager.getNumFieldBlocks(), 1); // if no field order is set this defaults to 1!

      std::vector<std::vector<std::string>> fieldOrder(3), fo_ut;
      fieldOrder[0].push_back("Ux");
      fieldOrder[0].push_back("Uy");
      fieldOrder[1].push_back("P");
      fieldOrder[2].push_back("rho");
      fieldOrder[2].push_back("T");
      dofManager.setFieldOrder(fieldOrder);
      dofManager.getFieldOrder(fo_ut);
      TEST_ASSERT(fieldOrder == fo_ut);
      TEST_EQUALITY(dofManager.getNumFieldBlocks(), 3); //

      TEST_ASSERT(dofManager.getElementBlock("left_lower_qtr") == connManager->getElementBlock("left_lower_qtr"));
      TEST_ASSERT(dofManager.getElementBlock("left_upper_qtr") == connManager->getElementBlock("left_upper_qtr"));
      TEST_ASSERT(dofManager.getElementBlock("right_half") == connManager->getElementBlock("right_half"));

      // check that each element is correct size
      std::vector<std::string> elementBlockIds;
      connManager->getElementBlockIds(elementBlockIds);
      for (std::size_t blk = 0; blk < connManager->numElementBlocks(); ++blk)
      {
         std::string blockId = elementBlockIds[blk];
         const std::vector<int> &elementBlock = connManager->getElementBlock(blockId);
         for (std::size_t elmt = 0; elmt < elementBlock.size(); ++elmt)
         {
            TEST_EQUALITY(connManager->getConnectivitySize(elementBlock[elmt]), 4);
         }
      }

      if (numProcs == 1)
      {
         TEST_EQUALITY(connManager->getNeighborElementBlock("left_lower_qtr").size(), 0);
         TEST_EQUALITY(connManager->getNeighborElementBlock("left_upper_qtr").size(), 0);
         TEST_EQUALITY(connManager->getNeighborElementBlock("right_half").size(), 0);
      }
      else
      {
         TEST_EQUALITY(connManager->getNeighborElementBlock("left_lower_qtr").size(), 1);
         TEST_EQUALITY(connManager->getNeighborElementBlock("left_upper_qtr").size(), 1);
         TEST_EQUALITY(connManager->getNeighborElementBlock("right_half").size(), 1);

         for (std::size_t blk = 0; blk < connManager->numElementBlocks(); ++blk)
         {
            const std::vector<int> &elementBlock = connManager->getNeighborElementBlock(elementBlockIds[blk]);
            for (std::size_t elmt = 0; elmt < elementBlock.size(); ++elmt)
            {
               TEST_EQUALITY(connManager->getConnectivitySize(elementBlock[elmt]), 0);
            }
         }
      }

      STKConnManager::GlobalOrdinal maxEdgeId = connManager->getMaxEntityId(connManager->getEdgeRank());
      STKConnManager::GlobalOrdinal nodeCount = connManager->getEntityCounts(connManager->getNodeRank());

      if (numProcs == 1)
      {
         const auto *conn1 = connManager->getConnectivity(1);
         const auto *conn2 = connManager->getConnectivity(2);

         /* GAH Note - Here is an example of testing the connectivity of the mesh in the exodus file.

This is just a start, to serve as an example. This has not been thought through carefully.

*/
         TEST_EQUALITY(conn1[0], 4);
         TEST_EQUALITY(conn1[1], 1);
         TEST_EQUALITY(conn1[2], 6);
         TEST_EQUALITY(conn1[3], 18);

         TEST_EQUALITY(conn2[0], 17);
         TEST_EQUALITY(conn2[1], 18);
         TEST_EQUALITY(conn2[2], 21);
         TEST_EQUALITY(conn2[3], 20);

         TEST_EQUALITY(conn1[3], conn2[1]);

         // Need to look at the exodus file and do a few of these sort of comparions
         //      TEST_EQUALITY(conn1[8],nodeCount+(maxEdgeId+1)+2);
         //      TEST_EQUALITY(conn2[8],nodeCount+(maxEdgeId+1)+3);
      }
      else
      {

         const auto *conn0 = connManager->getConnectivity(0);
         const auto *conn1 = connManager->getConnectivity(1);

         TEST_EQUALITY(conn0[0], 0 + myRank);
         TEST_EQUALITY(conn0[1], 1 + myRank);
         TEST_EQUALITY(conn0[2], 6 + myRank);
         TEST_EQUALITY(conn0[3], 5 + myRank);

         TEST_EQUALITY(conn1[0], 2 + myRank);
         TEST_EQUALITY(conn1[1], 3 + myRank);
         TEST_EQUALITY(conn1[2], 8 + myRank);
         TEST_EQUALITY(conn1[3], 7 + myRank);

         TEST_EQUALITY(conn0[8], nodeCount + (maxEdgeId + 1) + 1 + myRank);
         TEST_EQUALITY(conn1[8], nodeCount + (maxEdgeId + 1) + 3 + myRank);

         const auto *conn2 = connManager->getConnectivity(2); // this is the "neighbor element"
         const auto *conn3 = connManager->getConnectivity(3); // this is the "neighbor element"

         int otherRank = myRank == 0 ? 1 : 0;

         TEST_EQUALITY(conn2[0], 0 + otherRank);
         TEST_EQUALITY(conn2[1], 1 + otherRank);
         TEST_EQUALITY(conn2[2], 6 + otherRank);
         TEST_EQUALITY(conn2[3], 5 + otherRank);

         TEST_EQUALITY(conn3[0], 2 + otherRank);
         TEST_EQUALITY(conn3[1], 3 + otherRank);
         TEST_EQUALITY(conn3[2], 8 + otherRank);
         TEST_EQUALITY(conn3[3], 7 + otherRank);

         TEST_EQUALITY(conn2[8], nodeCount + (maxEdgeId + 1) + 1 + otherRank);
         TEST_EQUALITY(conn3[8], nodeCount + (maxEdgeId + 1) + 3 + otherRank);
      }
   }

   TEUCHOS_UNIT_TEST(AlbanyBlockedDOFManager_BlockedSTKDiscretizationTests, assortedTests)
   {

      // Set the static variable that denotes this as a Tpetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));

      Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();
      int numProcs = comm->getSize();
      int myRank = comm->getRank();

      // panzer::pauseToAttach();

      using Teuchos::RCP;
      using Teuchos::rcp;
      using Teuchos::rcp_dynamic_cast;

      // Build an STK Discretization object that holds the test mesh "2D_Blk_Test.e"

      // 2D, quad, 3 block mesh built in Cubit.
      Teuchos::RCP<Teuchos::ParameterList> discParams = rcp(new Teuchos::ParameterList);
      discParams->set<std::string>("Exodus Input File Name", "2D_Blk_Test.e");
      discParams->set<std::string>("Method", "Exodus");
      discParams->set<int>("Interleaved Ordering", 2);
      discParams->set<bool>("Use Composite Tet 10", 0);
      discParams->set<int>("Number Of Time Derivatives", 0);
      discParams->set<bool>("Use Serial Mesh", 1);

      // Need to test various meshes, with various element types and block structures.

      Teuchos::Array<Teuchos::RCP<AbstractSTKMeshStruct>> ms_array;

      ms_array.resize(3);

      Teuchos::RCP<Albany::AbstractMeshStruct> meshStruct0, meshStruct1, meshStruct2;
      meshStruct0 = DiscretizationFactory::createMeshStruct(discParams, comm, 0);
      meshStruct1 = DiscretizationFactory::createMeshStruct(discParams, comm, 0);
      meshStruct2 = DiscretizationFactory::createMeshStruct(discParams, comm, 0);

      discParams->set<std::string>("Field Order", "blocked: a");

      ms_array[0] = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct0);
      ms_array[1] = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct1);
      ms_array[2] = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct2);

      const AbstractFieldContainer::FieldContainerRequirements req;
      const Teuchos::RCP<StateInfoStruct> sis = Teuchos::rcp(new StateInfoStruct());
      const std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> side_set_sis;
      const std::map<std::string, AbstractFieldContainer::FieldContainerRequirements> side_set_req;

      ms_array[0]->setFieldAndBulkData(comm, discParams, 1, req, sis, AbstractMeshStruct::DEFAULT_WORKSET_SIZE);
      ms_array[1]->setFieldAndBulkData(comm, discParams, 3, req, sis, AbstractMeshStruct::DEFAULT_WORKSET_SIZE);
      ms_array[2]->setFieldAndBulkData(comm, discParams, 5, req, sis, AbstractMeshStruct::DEFAULT_WORKSET_SIZE);

      // Use the Albany STK interface as it is used elsewhere in the code
      auto stkDisc = Teuchos::rcp(new BlockedSTKDiscretization(discParams, ms_array, comm));
      stkDisc->updateMesh();

      auto nvs = stkDisc->getOverlapProductVectorSpace();
      TEST_EQUALITY(Teuchos::nonnull(nvs), true);

      auto nv0 = stkDisc->getOverlapVectorSpace(0);
      auto nv1 = stkDisc->getOverlapVectorSpace(1);
      auto nv2 = stkDisc->getOverlapVectorSpace(2);

      std::cout << " Dimension of the first block: " << nv0->dim() << std::endl;
      std::cout << " Dimension of the second block: " << nv1->dim() << std::endl;
      std::cout << " Dimension of the third block: " << nv2->dim() << std::endl;
      std::cout << " Dimension of the product vector space: " << nvs->dim() << std::endl;

      TEST_EQUALITY(nvs->dim(), nv0->dim() + nv1->dim() + nv2->dim());

      Teuchos::RCP<Thyra_BlockedLinearOp> bJacobianOp = stkDisc->createBlockedJacobianOp();

      Teuchos::RCP<const Thyra_LinearOp> jacobian00 = bJacobianOp->getBlock(0, 0);
      Teuchos::RCP<const Thyra_LinearOp> jacobian11 = bJacobianOp->getBlock(1, 1);
      Teuchos::RCP<const Thyra_LinearOp> jacobian22 = bJacobianOp->getBlock(2, 2);

      Teuchos::RCP<Teuchos::FancyOStream> out1 = Teuchos::VerboseObjectBase::getDefaultOStream();

      for (size_t i = 0; i < 3; ++i)
         for (size_t j = 0; j < i + 1; ++j)
         {
            Teuchos::RCP<const Thyra_LinearOp> jacobian = bJacobianOp->getBlock(i, j);

            *out1 << "Before describe jacobian " << i << " " << j << std::endl;
            Albany::describe(jacobian.getConst(), *out1, Teuchos::VERB_EXTREME);
            *out1 << "After describe jacobian " << i << " " << j << std::endl;
         }

      TEST_EQUALITY(Teuchos::nonnull(bJacobianOp), true);

      int fadl0 = stkDisc->getBlockFADLength(0);
      int fadl1 = stkDisc->getBlockFADLength(1);
      int fadl2 = stkDisc->getBlockFADLength(2);

      std::cout << " FAD length of the first block: " << fadl0 << std::endl;
      std::cout << " FAD length of the second block: " << fadl1 << std::endl;
      std::cout << " FAD length of the third block: " << fadl2 << std::endl;

      TEST_EQUALITY(fadl0, 4);
      TEST_EQUALITY(fadl1, 12);
      TEST_EQUALITY(fadl2, 20);

      int fado0 = stkDisc->getBlockFADOffset(0);
      int fado1 = stkDisc->getBlockFADOffset(1);
      int fado2 = stkDisc->getBlockFADOffset(2);

      std::cout << " FAD offset of the first block: " << fado0 << std::endl;
      std::cout << " FAD offset of the second block: " << fado1 << std::endl;
      std::cout << " FAD offset of the third block: " << fado2 << std::endl;

      TEST_EQUALITY(fado0, 0);
      TEST_EQUALITY(fado1, fadl0);
      TEST_EQUALITY(fado2, fadl0 + fadl1);
   }

   /*

   Note from GAH: Next steps. The Panzer package has quite a bit of testing of both the DOFManager and the Connection Manager.

   We can borrow a lot from the tests in :

   ~/Trilinos/packages/panzer/adapters-stk/test/stk_connmngr

   Indeed, the above draws pretty heavily from various tests in the Panzer package. I would continue along these lines given
   additional time.

   For the DOFManager and STKConnManager in Albany, I essentially "pulled in" much
   of Panzer's STK_Interface code and design.  This ended up being fairly
   straightforward and clean to do generally. The Panzer stk_interface is here:

   adapters-stk/src/

   I pulled over most of what I thought we might need; there is a lot there but I
   let the above tests guide much of what I did. I suspect most of the plumbing is
   here, but as I add tests I am still looking at the Panzer implementation for
   guidance (and to harvest from).

   */

}
