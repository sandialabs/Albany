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
#include <Omega_h_simplex.hpp> // Omega_h::simplex_degree
#include <Omega_h_for.hpp> // Omega_h::parallel_for
#include <Omega_h_array_ops.hpp> // Omega_h::get_sum
#include <Omega_h_atomics.hpp> // Omega_h::atomic_increment

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

Omega_h::Read<Omega_h::I8> markDownward(Omega_h::Mesh& mesh, Omega_h::Read<Omega_h::I8> isUpMarked, int upDim, int downDim) {
  REQUIRE(upDim-downDim==1);
  auto downEnts = mesh.ask_down(upDim,downDim).ab2b;
  Omega_h::Write<Omega_h::I8> downMarked(mesh.nents(downDim),0);
  const auto degree = Omega_h::simplex_degree(upDim, downDim);
  Omega_h::parallel_for(mesh.nents(upDim), OMEGA_H_LAMBDA(LO upEnt) {
    if(isUpMarked[upEnt]) {
      for(int j = 0; j < degree; j++) {
        const auto down = downEnts[upEnt*degree+j];
        //each down entity will be set multiple times, but the value being
        //written is the same so an atomic is not needed
        downMarked[down] = 1;
      }
    }
  });
  return downMarked;
}

void createTagFromClassification(Omega_h::Mesh& mesh, const int classId, const int classDim, const std::string tagName) {
  REQUIRE(classDim==1); //TODO support faces with classification?
  //add the edges with the given classification to the tag
  auto isinset = Omega_h::mark_by_class(&mesh, classDim, classDim, classId);
  mesh.add_tag(classDim, tagName, 1, isinset);
  //mark the vertices bounding the edges and add them to the tag
  auto lateralverts = markDownward(mesh, isinset, OMEGA_H_EDGE, OMEGA_H_VERT);
  mesh.add_tag(OMEGA_H_VERT, tagName, 1, lateralverts);
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
  //class_dim=1 on mesh edges by exo2osh.
  //A dimension and id uniquely defines a geometric model entity.
  const int lateralSide_classId = 1;
  const int lateralSide_classDim = 1;
  const auto lateralSide_name = "lateralside";
  createTagFromClassification(mesh, lateralSide_classId, lateralSide_classDim, lateralSide_name);

  auto conn_mgr = createOmegahConnManager(mesh, lateralSide_name, lateralSide_classDim);
  out << "Testing OmegahConnManager::partCtor()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getConnectivityMask)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahMesh(lib, "gis_unstruct_basal_populated.osh");
  //see above for discussion of tags and classification
  const int lateralSide_classId = 1;
  const int lateralSide_classDim = 1;
  const auto lateralSide_name = "lateralside";
  createTagFromClassification(mesh, lateralSide_classId, lateralSide_classDim, lateralSide_name); //490 verts

  //define tag for uppper half of lateral side
  const int upperSide_numVertsExpected = 238;
  const auto minYCoord = -1848.0;
  const int upperSide_classDim = 0;
  const auto upperSide_name = "upperside";
  auto vtxCoords = mesh.coords();
  Omega_h::Write<Omega_h::I8> isInSet_vtx(mesh.nents(OMEGA_H_VERT),0);
  const auto isLateralSide = mesh.get_array<Omega_h::I8>(OMEGA_H_EDGE, lateralSide_name);
  auto lateralVerts = markDownward(mesh, isLateralSide, OMEGA_H_EDGE, OMEGA_H_VERT);
  const auto isOwnedVtx = mesh.owned(OMEGA_H_VERT);
  Omega_h::parallel_for(mesh.nents(OMEGA_H_VERT), OMEGA_H_LAMBDA(LO vtx) {
      if(isOwnedVtx[vtx])
        isInSet_vtx[vtx] = ((vtxCoords[vtx*2+1] >= minYCoord) && lateralVerts[vtx]);
  });
  const int upperSide_numVerts = Omega_h::get_sum<Omega_h::I8>(mesh.library()->world(), isInSet_vtx);
  REQUIRE(upperSide_numVertsExpected == upperSide_numVerts);
  mesh.add_tag(upperSide_classDim, upperSide_name, 1, Omega_h::read(isInSet_vtx));

  auto conn_mgr = createOmegahConnManager(mesh, lateralSide_name, lateralSide_classDim);
  Teuchos::RCP<const panzer::FieldPattern> patternEdgeC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_LINE_C1_FEM<PHX::exec_space, double, double>>();
  conn_mgr->buildConnectivity(*patternEdgeC1);
  auto mask = conn_mgr->getConnectivityMask(upperSide_name);
  const int sum = std::accumulate(mask.begin(), mask.end(), 0);

  { //count the number of times upperSide vertices appear in the vertices bounding edges
    //this count should match the number of times '1' appears in the
    //getConnectivityMask(upperSide_name) call
    const auto lateralSide = mesh.get_array<Omega_h::I8>(OMEGA_H_EDGE, lateralSide_name);
    const auto isUpperSide = mesh.get_array<Omega_h::I8>(OMEGA_H_VERT, upperSide_name);
    auto edgeVerts = mesh.ask_down(OMEGA_H_EDGE,OMEGA_H_VERT).ab2b;
    const auto degree = Omega_h::simplex_degree(OMEGA_H_EDGE, OMEGA_H_VERT);
    assert(degree == 2);
    Omega_h::Write<Omega_h::LO> elmToUpperVtxCount_d(1,0);
    Omega_h::parallel_for(mesh.nents(OMEGA_H_EDGE), OMEGA_H_LAMBDA(LO edge) {
        if(lateralSide[edge]) {
          for(int i=0; i<degree; i++) {
            auto vtx = edgeVerts[edge*degree+i];
            if(isUpperSide[vtx]) Omega_h::atomic_increment(&elmToUpperVtxCount_d[0]);
          }
        }
    });
    const auto elmToUpperVtxCount = elmToUpperVtxCount_d.get(0);
    REQUIRE(elmToUpperVtxCount == sum);
  }
  out << "Testing OmegahConnManager::getConnectivityMask()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getConnectivityMask_box)
{
  const std::map<GO,std::array<GO,3>> elementGidToMask = {
    //{elmid, {dof mask}}   {vtx/dof gids}
    {0, {1, 0, 0}},   // {0, 7, 3},
    {1, {1, 0, 0}},   // {1, 3, 4},
    {2, {0, 1, 1}},   // {3, 1, 0},
    {3, {0, 0, 0}},   // {3, 6, 5},
    {4, {0, 1, 1}},   // {4, 2, 1},
    {5, {0, 0, 0}},   // {5, 4, 3},
    {6, {0, 0, 0}},   // {6, 3, 7},
    {7, {0, 0, 0}}    // {7, 8, 6}
  };

  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);


  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = createOmegahBoxMesh(lib);

  auto conn_mgr = createOmegahConnManager(mesh);
  Teuchos::RCP<const panzer::FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::exec_space, double, double>>();
  conn_mgr->buildConnectivity(*patternC1);

  const auto sideSetName = "leftSide";
  int dim = 0;
  auto vtxGids = conn_mgr->getGlobalDofNumbering(dim);
  Omega_h::Write<Omega_h::I8> isInSet_vtx(mesh.nents(dim));
  Omega_h::parallel_for(isInSet_vtx.size(), OMEGA_H_LAMBDA(LO i) {
      const auto gid = vtxGids[i];
      isInSet_vtx[i] = (gid >= 0 && gid <= 2) ? 1 : 0;
  });
  mesh.add_tag(dim, sideSetName, 1, Omega_h::read(isInSet_vtx));

  auto mask = conn_mgr->getConnectivityMask(sideSetName);

  const auto localElmIds = conn_mgr->getElementBlock("ignored");
  for( auto lid : localElmIds ) {
    auto elmGid = conn_mgr->getElementGlobalId(lid);
    const auto firstMaskIdx = lid*3;
    const std::array<GO,3> elmMask = {mask[firstMaskIdx], mask[firstMaskIdx+1], mask[firstMaskIdx+2]};
    const auto expectedMask = elementGidToMask.at(elmGid);
    if( expectedMask != elmMask ) {
      std::cerr << elmGid << " mask does not match expected mask\n";
    }
    REQUIRE( expectedMask == elmMask );
  }

  out << "Testing OmegahConnManager::getConnectivityMaskBox() on triangle mesh of square\n";
  success = true;
}
