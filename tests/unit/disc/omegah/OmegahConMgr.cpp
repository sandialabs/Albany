//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"
#include "OmegahConnManager.hpp"
#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahUtils.hpp"
#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

#include "Shards_CellTopology.hpp"

//need to test with field patterns
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"

#include <Omega_h_mark.hpp> // Omega_h::mark_by_class
#include <Omega_h_simplex.hpp> // Omega_h::simplex_degree
#include <Omega_h_for.hpp> // Omega_h::parallel_for
#include <Omega_h_array_ops.hpp> // Omega_h::get_sum
#include <Omega_h_atomics.hpp> // Omega_h::atomic_increment

#include <array> //std::array

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

struct PartSpecs {
  std::string name;
  Topo_type topo;
  int id;
};

Teuchos::RCP<Albany::OmegahGenericMesh>
createOmegahMesh(const std::string& filename,
                 const Teuchos::RCP<const Teuchos_Comm>& comm,
                 const std::vector<PartSpecs>& set_parts)
{
  REQUIRE(!filename.empty());

  auto pl = Teuchos::rcp(new Teuchos::ParameterList());
  pl->set<std::string>("Mesh Creation Method", "OshFile");
  pl->set("Input Filename",filename);
  Teuchos::Array<std::string> pnames;
  for (const auto& ps : set_parts) {
    pnames.push_back(ps.name);
    pl->sublist(ps.name).set("Topo",Albany::e2str(ps.topo));
    pl->sublist(ps.name).set("Id",ps.id);
  }
  pl->set("Mark Parts",pnames);
  return Teuchos::rcp(new Albany::OmegahGenericMesh(pl));
}

auto createOmegahConnManager(const Teuchos::RCP<Albany::OmegahGenericMesh>& mesh) {
  return Teuchos::rcp(new Albany::OmegahConnManager(mesh));
}

auto createOmegahConnManager(const Teuchos::RCP<Albany::OmegahGenericMesh>& mesh,
                             const std::string& partId) {
  return Teuchos::rcp(new Albany::OmegahConnManager(mesh, partId));
}

/* copied from tests/unit/disc/UnitTest_BlockedDOFManager.cpp */
template <template <typename,typename,typename> class Intrepid2Type>
Teuchos::RCP<const panzer::FieldPattern> buildFieldPattern()
{
  // build a geometric pattern from a single basis
  auto basis = Teuchos::rcp(new Intrepid2Type<PHX::exec_space, double, double>());
  auto pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
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

Omega_h::LO getNumOwnedEnts(Omega_h::Mesh& mesh, int dim) {
  REQUIRE(dim>=0 && dim <=3);
  auto owned = mesh.owned(dim);
  return Omega_h::get_sum(owned);
}

Omega_h::LO getNumOwnedElms(Omega_h::Mesh& mesh) {
  auto dim = mesh.dim();
  return getNumOwnedEnts(mesh,dim);
}

void checkOwnership(Omega_h::Mesh& mesh, const Albany::OmegahConnManager& connMgr) {
  const auto localElmIds = connMgr.getElementBlock();
  auto conMgrVtxGids = Omega_h::HostRead(connMgr.getGlobalDofNumbering(OMEGA_H_VERT));
  auto isVtxOwned = Omega_h::HostRead(mesh.owned(OMEGA_H_VERT));
  std::map<Omega_h::GO, Omega_h::I8> vtxGidOwned;
  for(int i = 0; i < conMgrVtxGids.size(); i++)
    vtxGidOwned[conMgrVtxGids[i]] = isVtxOwned[i];
  const auto partDim = connMgr.part_dim();
  int dofsPerElm = connMgr.getConnectivitySize(0);
  //this check only supports dofs at vertices
  if(partDim == 1) REQUIRE(dofsPerElm == 2);
  if(partDim == 2) REQUIRE(dofsPerElm == 3);
  if(partDim == 3) REQUIRE(dofsPerElm == 4);
  for( auto lid : localElmIds ) {
    auto dofGids = connMgr.getConnectivity(lid);
    auto dofOwned = connMgr.getOwnership(lid);
    for(int i=0; i<dofsPerElm; i++) {
      if(vtxGidOwned.at(dofGids[i])) {
        REQUIRE(Albany::Owned == dofOwned[i]);
      } else {
        REQUIRE(Albany::Ghosted == dofOwned[i]);
      }
    }
  }
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager1D)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh<1>(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mesh);
  out << "Testing OmegahConnManager constructor\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager1D_buildConnectivity)
{
  const std::map<GO,std::array<GO,2>> elementGidToDofs = {
    {0, {0, 1}},
    {1, {1, 2}}
  };

  auto teuchosComm = Albany::getDefaultComm();

  auto patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_LINE_C1_FEM>();

  auto mesh = createOmegahBoxMesh<1>(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mesh);
  conn_mgr->buildConnectivity(patternC1);
  REQUIRE(2 == conn_mgr->getConnectivitySize(0)); //all elements return the same size
  const auto localElmIds = conn_mgr->getElementBlock();
  const auto ohMesh = mesh->getOmegahMesh();
  if( ohMesh->comm()->size() == 4 && ohMesh->nelems() > 0 ) {
    REQUIRE(localElmIds.size() == 1);
  }
  for( auto lid : localElmIds ) {
    auto ptr = conn_mgr->getConnectivity(lid);
    auto elmGid = conn_mgr->getElementGlobalId(lid);
    const std::array<GO,2> dofs = {ptr[0], ptr[1]};
    const auto expectedDofs = elementGidToDofs.at(elmGid);
    REQUIRE( expectedDofs == dofs );
  }

  out << "Testing OmegahConnManager::buildConnectivity()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mesh);
  out << "Testing OmegahConnManager constructor\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManagerNoConnClone)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mesh);
  auto clone = conn_mgr->noConnectivityClone();
  out << "Testing OmegahConnManager::noConnectivityClone()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getElemsInBlock)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mesh);
  auto elmGids = conn_mgr->getElementsInBlock();
  int elmGidsSize = elmGids.size(); 
  REQUIRE(elmGidsSize == conn_mgr->getOwnedElementCount());
  auto numOwnedElms = getNumOwnedElms(*mesh->getOmegahMesh());
  REQUIRE(numOwnedElms == conn_mgr->getOwnedElementCount());
  out << "Testing OmegahConnManager::getElementsInBlock()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getBlockId)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
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
  auto teuchosComm = Albany::getDefaultComm();

  auto mesh = createOmegahBoxMesh(teuchosComm);
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

  auto teuchosComm = Albany::getDefaultComm();

  auto patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM>();

  auto mesh = createOmegahBoxMesh(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mesh);
  conn_mgr->buildConnectivity(patternC1);
  REQUIRE(3 == conn_mgr->getConnectivitySize(0)); //all elements return the same size
  const auto localElmIds = conn_mgr->getElementBlock();
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
  auto teuchosComm = Albany::getDefaultComm();

  // The 'lateralside' side set in the exodus file is given class_id=1 and
  // class_dim=1 on mesh edges by exo2osh.
  const int lateralSide_classId = 1;
  const auto lateralSide_name = "lateralside";

  std::vector<PartSpecs> lateralSide = {
    { lateralSide_name, Topo_type::edge, lateralSide_classId }
  };
  auto mesh = createOmegahMesh("gis_unstruct_basal_populated.osh",teuchosComm, lateralSide);

  auto conn_mgr = createOmegahConnManager(mesh, lateralSide_name);
  out << "Testing OmegahConnManager::partCtor()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getConnectivityMask)
{
  auto teuchosComm = Albany::getDefaultComm();

  // The 'lateralside' side set in the exodus file is given class_id=1 and
  // class_dim=1 on mesh edges by exo2osh.
  const int lateralSide_classId = 1;
  const auto lateralSide_name = "lateralside";

  std::vector<PartSpecs> lateralSide = {
    { lateralSide_name, Topo_type::edge, lateralSide_classId }
  };
  auto albanyMesh = createOmegahMesh("gis_unstruct_basal_populated.osh",teuchosComm, lateralSide);
  auto mesh = albanyMesh->getOmegahMesh();

  //define tag for uppper half of lateral side
  const int upperSide_numVertsExpected = 238;
  const auto minYCoord = -1848.0;
  const int upperSide_classDim = 0;
  const auto upperSide_name = "upperside";
  auto vtxCoords = mesh->coords();
  Omega_h::Write<Omega_h::I8> isInSet_vtx(mesh->nents(OMEGA_H_VERT),0);
  const auto isLateralSide = mesh->get_array<Omega_h::I8>(OMEGA_H_EDGE, lateralSide_name);
  auto lateralVerts = markDownward(*mesh, isLateralSide, OMEGA_H_EDGE, OMEGA_H_VERT);
  const auto isOwnedVtx = mesh->owned(OMEGA_H_VERT);
  Omega_h::parallel_for(mesh->nents(OMEGA_H_VERT), OMEGA_H_LAMBDA(LO vtx) {
      if(isOwnedVtx[vtx])
        isInSet_vtx[vtx] = ((vtxCoords[vtx*2+1] >= minYCoord) && lateralVerts[vtx]);
  });
  const int upperSide_numVerts = Omega_h::get_sum<Omega_h::I8>(mesh->library()->world(), isInSet_vtx);
  REQUIRE(upperSide_numVertsExpected == upperSide_numVerts);
  mesh->add_tag(upperSide_classDim, upperSide_name, 1, Omega_h::read(isInSet_vtx));

  auto conn_mgr = createOmegahConnManager(albanyMesh, lateralSide_name);
  auto patternEdgeC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_LINE_C1_FEM>();
  conn_mgr->buildConnectivity(patternEdgeC1);
  auto mask = conn_mgr->getConnectivityMask(upperSide_name);
  const int sum = std::accumulate(mask.begin(), mask.end(), 0);

  { //count the number of times upperSide vertices appear in the vertices bounding edges
    //this count should match the number of times '1' appears in the
    //getConnectivityMask(upperSide_name) call
    const auto lateralSide = mesh->get_array<Omega_h::I8>(OMEGA_H_EDGE, lateralSide_name);
    const auto isUpperSide = mesh->get_array<Omega_h::I8>(OMEGA_H_VERT, upperSide_name);
    auto edgeVerts = mesh->ask_down(OMEGA_H_EDGE,OMEGA_H_VERT).ab2b;
    const auto degree = Omega_h::simplex_degree(OMEGA_H_EDGE, OMEGA_H_VERT);
    assert(degree == 2);
    Omega_h::Write<Omega_h::LO> elmToUpperVtxCount_d(1,0);
    Omega_h::parallel_for(mesh->nents(OMEGA_H_EDGE), OMEGA_H_LAMBDA(LO edge) {
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

  const std::map<GO,std::array<GO,3>> elementGidToDofGids = {
    //{elmid, {vtx/dof gids}}
    {0, {0, 7, 3}},
    {1, {1, 3, 4}},
    {2, {3, 1, 0}},
    {3, {3, 6, 5}},
    {4, {4, 2, 1}},
    {5, {5, 4, 3}},
    {6, {6, 3, 7}},
    {7, {7, 8, 6}}
  };

  auto teuchosComm = Albany::getDefaultComm();

  auto albanyMesh = createOmegahBoxMesh(teuchosComm);
  auto& mesh = albanyMesh->getOmegahMesh();

  auto conn_mgr = createOmegahConnManager(albanyMesh);
  auto patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM>();
  conn_mgr->buildConnectivity(patternC1);

  const auto sideSetName = "leftSide";
  int dim = 0;
  auto vtxGids = conn_mgr->getGlobalDofNumbering(dim);
  Omega_h::Write<Omega_h::I8> isInSet_vtx(mesh->nents(dim));
  Omega_h::parallel_for(isInSet_vtx.size(), OMEGA_H_LAMBDA(LO i) {
      const auto gid = vtxGids[i];
      isInSet_vtx[i] = (gid >= 0 && gid <= 2) ? 1 : 0;
  });
  mesh->add_tag(dim, sideSetName, 1, Omega_h::read(isInSet_vtx));

  auto mask = conn_mgr->getConnectivityMask(sideSetName);

  const auto localElmIds = conn_mgr->getElementBlock("ignored");
  for( auto lid : localElmIds ) {
    auto elmGid = conn_mgr->getElementGlobalId(lid);
    auto dofGids = conn_mgr->getConnectivity(lid);
    const auto firstMaskIdx = lid*3;
    const std::array<GO,3> elmDofGids = {dofGids[0], dofGids[1], dofGids[2]};
    const auto expectedDofGids = elementGidToDofGids.at(elmGid);
    if( expectedDofGids != elmDofGids ) {
      std::cerr << elmGid << " does not contain the expected DOF global ids\n";
    }
    REQUIRE( expectedDofGids == elmDofGids );
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

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_buildConnectivityOwnership)
{
  auto teuchosComm = Albany::getDefaultComm();

  auto patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM>();

  auto mesh = createOmegahMesh("gis_unstruct_basal_populated.osh",teuchosComm,{});
  auto conn_mgr = createOmegahConnManager(mesh);
  conn_mgr->buildConnectivity(patternC1);
  checkOwnership(*mesh->getOmegahMesh(),*conn_mgr);
  out << "Testing OmegahConnManager::buildConnectivityOwnership()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_buildPartConnectivityOwnership)
{
  auto teuchosComm = Albany::getDefaultComm();

  const auto lateralSide_name = "lateralside";
  const int lateralSide_classId = 1;
  std::vector<PartSpecs> lateralSide = {
    { "lateralside", Topo_type::edge, lateralSide_classId }
  };

  auto mesh = createOmegahMesh("gis_unstruct_basal_populated.osh",teuchosComm,{lateralSide});

  auto conn_mgr = createOmegahConnManager(mesh, lateralSide_name);

  auto patternEdgeC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_LINE_C1_FEM>();
  conn_mgr->buildConnectivity(patternEdgeC1);
  checkOwnership(*mesh->getOmegahMesh(),*conn_mgr);
  out << "Testing OmegahConnManager::buildPartConnectivityOwnership()\n";
  success = true;
}
