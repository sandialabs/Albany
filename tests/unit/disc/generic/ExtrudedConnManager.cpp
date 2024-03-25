//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_UnitTestSession.hpp"
#include "DummyConnManager.hpp"
#include "Albany_ExtrudedConnManager.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "DummyMesh.hpp"


#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_ElemFieldPattern.hpp>
#include <Intrepid2_HierarchicalBasisFamily.hpp>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <random>

Teuchos::RCP<panzer::FieldPattern>
create_agg_fp (const std::vector<Teuchos::RCP<panzer::FieldPattern>>& fps)
{
  std::vector<std::pair<panzer::FieldType,Teuchos::RCP<const panzer::FieldPattern>>> tmp;
  std::vector<std::tuple< int, panzer::FieldType, Teuchos::RCP<const panzer::FieldPattern> > > faConstruct;
  const auto CG = panzer::FieldType::CG;
  for (std::size_t i=0; i<fps.size(); ++i) {
    tmp.push_back(std::make_pair(CG,fps[i]));
    faConstruct.emplace_back(i, CG, fps[i]);
  }
  auto ga_fp = Teuchos::rcp(new panzer::GeometricAggFieldPattern(tmp));
  return Teuchos::rcp(new panzer::FieldAggPattern(faConstruct, ga_fp));
}

TEUCHOS_UNIT_TEST(ConnMgrTensProd, 3D)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto& ts = UnitTestSession::instance();
  std::cout << "seed: " << ts.rng_seed << "\n";

  Teuchos::Array<int> nelems(3);
  nelems[0] = 4;
  nelems[1] = 3;
  nelems[2] = 2;

  // 1d, 2d, and 3d topologies
  auto line  = shards::getCellTopologyData<shards::Line<2>>();
  auto tria  = shards::getCellTopologyData<shards::Triangle<3>>();
  auto wedge = shards::getCellTopologyData<shards::Wedge<6>>();

  // Create an intrepid2 tensor basis
  using basis_family_type = Intrepid2::HierarchicalBasisFamily<PHX::Device>;
  using tria_basis_type  = typename basis_family_type::HGRAD_TRI;
  using line_basis_type  = typename basis_family_type::HGRAD_LINE;
  using wedge_basis_type = typename basis_family_type::HGRAD_WEDGE;
  using hexa_basis_type  = typename basis_family_type::HGRAD_HEX;

  auto line_basis  = Teuchos::rcp(new line_basis_type(1));
  auto tria_basis  = Teuchos::rcp(new tria_basis_type(1));
  auto wedge_basis = Teuchos::rcp(new wedge_basis_type(1,1));
  auto wedge_p2_basis = Teuchos::rcp(new wedge_basis_type(2,2));
  auto hexa_basis  = Teuchos::rcp(new hexa_basis_type(1));

  // Create Intrepid2FieldPatterns
  auto fp_1d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(line_basis));
  auto fp_2d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(tria_basis));
  auto fp_3d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_basis));
  auto fp_3d_p2 = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_p2_basis));
  auto fp_3d_hex = Teuchos::rcp(new panzer::Intrepid2FieldPattern(hexa_basis));

  // Build 1d, 2d, and 3d conn managers
  auto conn_mgr_line  = Teuchos::rcp(new DummyConnManager(line));
  auto conn_mgr_tria  = Teuchos::rcp(new DummyConnManager(tria));
  auto conn_mgr_wedge = Teuchos::rcp(new DummyConnManager(wedge));

  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine;

  // Create dummy basal mesh
  auto basal_mesh = Teuchos::rcp(new DummyMesh2d(ipdf(10,20)(engine)));
  
  for (auto ordering : {LayeredMeshOrdering::COLUMN,LayeredMeshOrdering::COLUMN}) {

    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set<int>("NumLayers",0);
    params->set("Columnwise Ordering", ordering==LayeredMeshOrdering::COLUMN);

    // Test ExtrudedConnManager exception handling in constructor
    TEST_THROW (Teuchos::rcp(new DummyExtrudedMesh(Teuchos::null,params,comm)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new DummyExtrudedMesh(basal_mesh,Teuchos::null,comm)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new DummyExtrudedMesh(basal_mesh,params,comm)),
                std::invalid_argument);

    ipdf nl_pdf (1,5);
    params->set<int>("NumLayers",nl_pdf(engine));
    auto extruded_mesh = Teuchos::rcp(new DummyExtrudedMesh(basal_mesh,params,comm));

    // Test ExtrudedConnManager exception handling in constructor
    TEST_THROW (Teuchos::rcp(new ExtrudedConnManager(Teuchos::null,extruded_mesh)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,Teuchos::null)),
                std::invalid_argument);

    // Build extruded conn manager
    auto conn_mgr_ext = Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,extruded_mesh));

    auto elem_fp = Teuchos::rcp(new panzer::ElemFieldPattern(wedge));
    auto fp_3d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_basis));
    auto fp_2d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(tria_basis));

    // Bad field agg pattern: contains non-intrepid patterns
    auto bad_fp1 = create_agg_fp({elem_fp});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp1),std::runtime_error);

    // Bad field agg pattern: contains different intrepid patterns
    auto bad_fp2 = create_agg_fp({fp_3d,fp_3d_p2});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp2),std::runtime_error);

    // Bad field agg pattern: contains patterns with wrong cell topo
    auto bad_fp3 = create_agg_fp({fp_2d});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp3),std::runtime_error);

    // Bad field agg pattern: basis is tens prod with wrong basal basis topology
    auto bad_fp4 = create_agg_fp({fp_3d_hex});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp4),std::invalid_argument);
  }
}
