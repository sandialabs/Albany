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

  // 1d, 2d, and 3d topologies
  auto line  = shards::getCellTopologyData<shards::Line<2>>();
  auto tria  = shards::getCellTopologyData<shards::Triangle<3>>();
  auto wedge = shards::getCellTopologyData<shards::Wedge<6>>();

  // Create an intrepid2 tensor basis
  using basis_family_type = Intrepid2::HierarchicalBasisFamily<PHX::Device>;
  using tria_basis_type  = typename basis_family_type::HGRAD_TRI;
  using line_basis_type  = typename basis_family_type::HGRAD_LINE;
  using wedge_basis_type = typename basis_family_type::HGRAD_WEDGE;
  using wedge_basis_mono_type = Intrepid2::Basis_HGRAD_WEDGE_C1_FEM<PHX::Device,RealType,RealType>;
  using hexa_basis_type  = typename basis_family_type::HGRAD_HEX;

  auto line_basis  = Teuchos::rcp(new line_basis_type(1));
  auto tria_basis  = Teuchos::rcp(new tria_basis_type(1));
  auto wedge_basis = Teuchos::rcp(new wedge_basis_type(1,2));
  auto wedge_p2_basis = Teuchos::rcp(new wedge_basis_type(2,2));
  auto wedge_mono_basis = Teuchos::rcp(new wedge_basis_mono_type());
  auto hexa_basis  = Teuchos::rcp(new hexa_basis_type(1));

  // Create field patterns for testing
  auto fp_1d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(line_basis));
  auto fp_2d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(tria_basis));
  auto fp_3d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_mono_basis));
  auto fp_3d_tens = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_basis));
  auto fp_3d_p2 = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_p2_basis));
  auto fp_3d_hex = Teuchos::rcp(new panzer::Intrepid2FieldPattern(hexa_basis));
  auto elem_fp = Teuchos::rcp(new panzer::ElemFieldPattern(wedge));

  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine;
  ipdf nlay_pdf (1,5);
  ipdf ntri_pdf (5,10);

  // Create dummy basal mesh
  GO ne_x_lcl = 1;//ntri_pdf(engine);
  auto numLayers = 1;//nlay_pdf(engine);

  // Basal mesh and monolithic (not extruded) 3d mesh
  auto mesh_2d = Teuchos::rcp(new DummyMesh(ne_x_lcl,comm));

  GO num_glb_tria;
  Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&ne_x_lcl,&num_glb_tria);

  std::cout << "RUNNING TESTS WITH:\n"
            << "  num 2d elems: " << 2*ne_x_lcl << "\n"
            << "  num layers  : " << numLayers << "\n";

  for (auto ordering : {LayeredMeshOrdering::COLUMN,LayeredMeshOrdering::LAYER}) {
    // Build 2d and 3d conn managers (not extruded)
    auto conn_mgr_tria  = Teuchos::rcp(new DummyConnManager(mesh_2d));

    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set<int>("NumLayers",0);
    params->set<int>("Workset Size",1000);
    params->set("Columnwise Ordering", ordering==LayeredMeshOrdering::COLUMN);

    // Bad pointers
    TEST_THROW (Teuchos::rcp(new DummyExtrudedMesh(Teuchos::null,params,comm)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new DummyExtrudedMesh(mesh_2d,Teuchos::null,comm)),
                std::invalid_argument);

    // Bad num layers
    TEST_THROW (Teuchos::rcp(new DummyExtrudedMesh(mesh_2d,params,comm)),
                Teuchos::Exceptions::InvalidParameterValue);

    params->set<int>("NumLayers",numLayers);
    auto extruded_mesh = Teuchos::rcp(new DummyExtrudedMesh(mesh_2d,params,comm));

    // Bad pointers
    TEST_THROW (Teuchos::rcp(new ExtrudedConnManager(Teuchos::null,extruded_mesh)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,Teuchos::null)),
                std::invalid_argument);

    // Build extruded conn manager
    auto conn_mgr_ext = Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,extruded_mesh));

    // Bad field agg pattern: contains non-intrepid patterns
    auto bad_fp1 = create_agg_fp({elem_fp});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp1),std::bad_cast);

    // Bad field agg pattern: contains different intrepid patterns
    auto bad_fp2 = create_agg_fp({fp_3d,fp_3d_p2});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp2),std::runtime_error);

    // Bad field agg pattern: contains patterns with wrong cell topo
    auto bad_fp3 = create_agg_fp({fp_2d});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp3),std::runtime_error);

    // Bad field agg pattern: basis is tens prod with wrong basal basis topology
    auto bad_fp4 = create_agg_fp({fp_3d_hex});
    TEST_THROW (conn_mgr_ext->buildConnectivity(*bad_fp4),std::invalid_argument);

    // Create extruded and monolithic connectivities
    auto fp_wedge_tens = create_agg_fp({fp_3d_tens});
    conn_mgr_ext->buildConnectivity(*fp_wedge_tens);

    auto mesh_3d = Teuchos::rcp(new DummyMesh(ne_x_lcl,numLayers,ordering,comm));
    auto conn_mgr_wedge = Teuchos::rcp(new DummyConnManager(mesh_3d));
    auto fp_wedge = create_agg_fp({fp_3d});
    conn_mgr_wedge->buildConnectivity(*fp_wedge);

    // Compare connectivities
    auto layers_data_lid = extruded_mesh->layers_data_lid();

    for (int icol=0; icol<mesh_2d->get_num_local_elements(); ++icol) {
      auto belem = conn_mgr_tria->getElementsInBlock()[icol];
      for (int ilev=0; ilev<numLayers; ++ilev) {
        int ie = layers_data_lid->getId(icol,ilev);
        auto cnt = conn_mgr_ext->getConnectivitySize(ie);
        auto conn3d = conn_mgr_wedge->getConnectivity(ie);
        auto conn = conn_mgr_ext->getConnectivity(ie);
        // TEST_EQUALITY_CONST( cnt, 6 );

        std::cout << "ie=" << ie << ", icol=" << icol << ", ilev=" << ilev << "\n";
        std::cout << " conn size: " << cnt << "\n";
        std::stringstream act,w3d;
        act << " actual conn:";
        w3d << " mono3d conn:";
        for (int idof=0; idof<3; ++idof) {
          act << " " << conn[idof];
          w3d << " " << conn3d[idof];
          // TEST_EQUALITY (conn[idof],gnode);
        }
        for (int idof=3; idof<6; ++idof) {
          act << " " << conn[idof];
          w3d << " " << conn3d[idof];
          // TEST_EQUALITY (conn[3+idof],gnode);
        }
        std::cout << act.str() << "\n";
        std::cout << w3d.str() << "\n";
      }
    }
  }
}
