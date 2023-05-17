//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"
#include "OmegahConnManager.hpp"
#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

#include "Shards_CellTopology.hpp"

//need to test with field patterns
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"

#include <Omega_h_build.hpp> // Omega_h::build_box
#include <Omega_h_file.hpp> // Omega_h::binary::read
#include <Omega_h_mark.hpp> // Omega_h::mark_by_class
#include <Omega_h_simplex.hpp> // Omega_h::simplex_down_template

#include <array> //std::array

#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

Omega_h::Mesh createOmegahBoxMesh(Omega_h::Library& lib) {
  return Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0, false);
}

Omega_h::Mesh createOmegahMesh(Omega_h::Library& lib, std::string name) {
  REQUIRE(!name.empty());
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(name, lib.world(), &mesh);
  mesh.balance(); // re-partition to the number of ranks in world communicator
  return mesh;
}

auto createOmegahConnManager(Omega_h::Mesh& mesh) {
  return Teuchos::rcp(new Albany::OmegahConnManager(mesh));
}

auto createOmegahConnManager(Omega_h::Mesh& mesh, std::string partId, const int partDim) {
  return Teuchos::rcp(new Albany::OmegahConnManager(mesh, partId, partDim));
}

/* copied from tests/unit/disc/UnitTest_BlockedDOFManager.cpp */
template <typename Intrepid2Type>
Teuchos::RCP<const panzer::FieldPattern> buildFieldPattern()
{
  // build a geometric pattern from a single basis
  Teuchos::RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis = Teuchos::rcp(new Intrepid2Type);
  Teuchos::RCP<const panzer::FieldPattern> pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  return pattern;
}


TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);
  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);
  auto conn_mgr = createOmegahConnManager(mesh);
  out << "Testing OmegahConnManager constructor\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManagerNoConnClone)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);
  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);
  auto conn_mgr = createOmegahConnManager(mesh);
  auto clone = conn_mgr->noConnectivityClone();
  out << "Testing OmegahConnManager::noConnectivityClone()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getElemsInBlock)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);
  auto conn_mgr = createOmegahConnManager(mesh);
  auto elmGids = conn_mgr->getElementsInBlock();
  REQUIRE(elmGids.size() == conn_mgr->getOwnedElementCount());
  out << "Testing OmegahConnManager::getElementsInBlock()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getBlockId)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);
  auto conn_mgr = createOmegahConnManager(mesh);
  std::vector<std::string> blockIds;
  conn_mgr->getElementBlockIds(blockIds);
  REQUIRE(conn_mgr->getBlockId(0) == blockIds[0]);
  const auto nelms = conn_mgr->getOwnedElementCount();
  REQUIRE(blockIds[0] == conn_mgr->getBlockId(nelms-1));
  out << "Testing OmegahConnManager::getBlockId()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getBlockTopologies)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);
  auto conn_mgr = createOmegahConnManager(mesh);
  std::vector<shards::CellTopology> topoTypes;
  conn_mgr->getElementBlockTopologies(topoTypes);
  shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
  REQUIRE(triTopo == topoTypes[0]);
  out << "Testing OmegahConnManager::getElementBlockTopologies()\n";
  success = true;
}


TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_buildConnectivity)
{
  const std::map<GO,std::array<GO,3>> elementGidToDofs = {
    {0, {0, 7, 3}},
    {1, {1, 3, 4}},
    {2, {3, 1, 0}},
    {3, {3, 6, 5}},
    {4, {4, 2, 1}},
    {5, {5, 4, 3}},
    {6, {6, 3, 7}},
    {7, {7, 8, 6}}
  };

  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  Teuchos::RCP<const panzer::FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::exec_space, double, double>>();

  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);
  auto conn_mgr = createOmegahConnManager(mesh);
  conn_mgr->buildConnectivity(*patternC1);
  REQUIRE(3 == conn_mgr->getConnectivitySize(0)); //all elements return the same size
  const auto localElmIds = conn_mgr->getElementBlock("ignored");
  for( auto lid : localElmIds ) {
    auto ptr = conn_mgr->getConnectivity(lid);
    auto elmGid = conn_mgr->getElementGlobalId(lid);
    const std::array<GO,3> dofs = {ptr[0], ptr[1], ptr[2]};
    const auto expectedDofs = elementGidToDofs.at(elmGid);
    REQUIRE( expectedDofs == dofs );
  }
  out << "Testing OmegahConnManager::buildConnectivity()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_checkTetrahedronCanonicalEntityOrder)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto tetFacePermOmegah2Shards = {3,0,1,2};
  auto triVtxPermOmegah2Shards = {{0,1,2},
                                  {0,1,2},
                                  {0,1,2},
                                  {2,0,1}}; //face 2 is the only one which has a different order

  const int elem_dim = 3;
  const int bdry_dim = 2;
  const int vtx_dim = 0;
  for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
    for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
      int which_vert = 0;
      auto ohIdx = Omega_h::simplex_down_template(elem_dim, bdry_dim, bdry, vert);

      shards::CellTopology tetTopo(shards::getCellTopologyData< shards::Tetrahedron<4> >());
      auto shIdx = tetTopo.getNodeMap(bdry_dim,bdry,vert);

      //if(ohIdx != shIdx) {
        std::cerr << "\ntet: bdry_dim which_bdry which_vert:"
                  << bdry_dim << " " << bdry << " "<< vert
                  << " Omega_h index " << ohIdx << " Shards index " << shIdx << "\n";
      //}
      //REQUIRE(ohIdx == shIdx);
    }
  }

  out << "Testing OmegahConnManager:: canonical entity order - Omega_h vs Shards()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_partCtor)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahMesh(lib, "gis_unstruct_basal_populated.osh");
  //The omegah 'exo2osh' converter creates geometric model entities from node
  //and side sets that exist within the exodus file.
  //The mesh entities in the sets are then 'classified' (sets the association)
  //on those model entities.
  //'Classification' of mesh entities to the geometric model is an alternative
  //to the generic creation of 'parts' (sets of mesh entities with a label) and
  //provides a subset of the functionality.
  //Note, 'classification' is the approach taken when having (at a minimum) a
  //topological definition of the domain is a common part of the mesh
  //generation/adaptation workflow.
  //The 'lateralside' side set in the exodus file is given class_id=1 and
  //class_dim=1 by exo2osh.
  //A dimension and id uniquely defines a geometric model entity.
  const int lateralSide_classId = 1;
  const int lateralSide_classDim = 1;
  const auto lateralSide_name = "lateralside";
  for(int dim=0; dim<=lateralSide_classDim; dim++) {
    auto isInSet = Omega_h::mark_by_class(&mesh, dim,
        lateralSide_classDim, lateralSide_classId);
    mesh.add_tag(dim, lateralSide_name, 1, isInSet);
  }

  auto conn_mgr = createOmegahConnManager(mesh, lateralSide_name, lateralSide_classDim);
  out << "Testing OmegahConnManager::contains()\n";
  success = true;
}
