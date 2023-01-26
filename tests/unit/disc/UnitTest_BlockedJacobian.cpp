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
   RCP<const panzer::FieldPattern> buildFieldPattern()
   {
      // build a geometric pattern from a single basis
      RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis = rcp(new Intrepid2Type);
      RCP<const panzer::FieldPattern> pattern = rcp(new panzer::Intrepid2FieldPattern(basis));
      return pattern;
   }

   TEUCHOS_UNIT_TEST(AlbanyBlockedDOFManager_BlockedJacobian, assortedTests)
   {

      // Set the static variable that denotes this as a Tpetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));

      RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

      // panzer::pauseToAttach();

      RCP<Teuchos::ParameterList> discParams = rcp(new Teuchos::ParameterList);

      std::string sideName;

      discParams->set<std::string>("Gmsh Input Mesh File Name", "cube.msh");
      discParams->set<std::string>("Method", "Gmsh");

      sideName = "BoundarySideSet_Bottom";

      discParams->set<int>("Interleaved Ordering", 2);
      discParams->set<bool>("Use Composite Tet 10", 0);
      discParams->set<int>("Number Of Time Derivatives", 0);
      discParams->set<bool>("Use Serial Mesh", 1);

      RCP<Teuchos::ParameterList> ssDiscParams = Teuchos::sublist(discParams, "Side Set Discretizations", false);
      Teuchos::Array<std::string> sideset(1);
      sideset[0] = sideName;
      ssDiscParams->set<Teuchos::Array<std::string>>("Side Sets", sideset);
      RCP<Teuchos::ParameterList> params_ss = Teuchos::sublist(ssDiscParams, sideset[0], false);
      params_ss->set<std::string>("Method", "SideSetSTK");

      RCP<Albany::AbstractMeshStruct> meshStruct = DiscretizationFactory::createMeshStruct(discParams, comm, 0);

      // Blocked discretization:
      {
         RCP<Teuchos::ParameterList> blockedDiscParams = rcp(new Teuchos::ParameterList);
         RCP<Teuchos::ParameterList> mParams = Teuchos::sublist(blockedDiscParams, "Mesh", false);
         mParams->set<std::string>("Name", "Body 1");
         mParams->set<std::string>("Type", "Extruded");
         RCP<Teuchos::ParameterList> smParams = Teuchos::sublist(mParams, "Side Meshes", false);
         smParams->set<std::string>("Sidesets", "[Bottom,Top]");

         RCP<Teuchos::ParameterList> dParams = Teuchos::sublist(blockedDiscParams, "Discretization", false);
         dParams->set<int>("Num Blocks", 3);

         RCP<Teuchos::ParameterList> db0Params = Teuchos::sublist(dParams, "Block 0", false);
         RCP<Teuchos::ParameterList> db1Params = Teuchos::sublist(dParams, "Block 1", false);
         RCP<Teuchos::ParameterList> db2Params = Teuchos::sublist(dParams, "Block 2", false);

         db0Params->set<std::string>("Name", "vol_C1");
         db0Params->set<std::string>("Mesh", "ElementBlock_Body_1");
         db0Params->set<std::string>("FE Type", "HGRAD_C1");
         db0Params->set<std::string>("Domain", "Volume");

         RCP<Teuchos::ParameterList> sParams = Teuchos::sublist(blockedDiscParams, "Solution", false);
         sParams->set<std::string>("blocks names", "[ [Ux], [Uy], [Uz], [N], [h] ]");
         sParams->set<std::string>("blocks discretizations", "[ [vol_C1], [vol_C1], [vol_C1], [vol_C1], [vol_C1] ]");

         RCP<AbstractSTKMeshStruct> ms = rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct);

         const RCP<StateInfoStruct> sis = rcp(new StateInfoStruct());
         const std::map<std::string, RCP<Albany::StateInfoStruct>> side_set_sis;

         ms->setFieldAndBulkData(comm, sis, meshStruct->getMeshSpecs()[0]->worksetSize);

         // Use the Albany STK interface as it is used elsewhere in the code
         auto stkDisc = rcp(new BlockedSTKDiscretization(blockedDiscParams, ms, comm));
         stkDisc->updateMesh();

         RCP<Thyra_BlockedLinearOp> bJacobianOp = stkDisc->createBlockedJacobianOp();

         for (int i = 0; i < stkDisc->getNumFieldBlocks(); ++i)
            for (int j = 0; j < stkDisc->getNumFieldBlocks(); ++j)
            {
               RCP<const Thyra_LinearOp> jacobian = bJacobianOp->getBlock(i, j);

               std::ofstream myfile;
               RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(rcpFromRef(myfile));
               myfile.open("b_jac_" + std::to_string(i) + "_" + std::to_string(j) + ".txt");
               Albany::describe(jacobian.getConst(), *fancy, Teuchos::VERB_EXTREME);
               myfile.close();
            }
      }

      //Non-block discretization:
      {
         RCP<Teuchos::ParameterList> dDiscParams = rcp(new Teuchos::ParameterList);
         RCP<Teuchos::ParameterList> mParams = Teuchos::sublist(dDiscParams, "Mesh", false);
         mParams->set<std::string>("Name", "Body 1");
         mParams->set<std::string>("Type", "Extruded");

         int neq = 5;

         RCP<Albany::AbstractMeshStruct> meshStruct = DiscretizationFactory::createMeshStruct(discParams, comm, 0);
         RCP<AbstractSTKMeshStruct> ms = rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct);

         const RCP<StateInfoStruct> sis = rcp(new StateInfoStruct());
         const std::map<std::string, RCP<Albany::StateInfoStruct>> side_set_sis;

         ms->setFieldAndBulkData(comm, sis, meshStruct->getMeshSpecs()[0]->worksetSize);

         // Use the Albany STK interface as it is used elsewhere in the code
         auto stkDisc = rcp(new STKDiscretization(dDiscParams, neq, ms, comm));
         stkDisc->updateMesh();

         RCP<Thyra_LinearOp> jacobian = stkDisc->createJacobianOp();

         std::ofstream myfile;
         RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(rcpFromRef(myfile));
         myfile.open("jac.txt");
         Albany::describe(jacobian.getConst(), *fancy, Teuchos::VERB_EXTREME);
         myfile.close();
      }
   }
}
