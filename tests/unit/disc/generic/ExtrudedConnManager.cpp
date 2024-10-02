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

TEUCHOS_UNIT_TEST(ExtrudedConnMgr, Exceptions)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto& ts = UnitTestSession::instance();

  // 2d, and 3d topologies
  auto wedge = shards::getCellTopologyData<shards::Wedge<6>>();

  // Create an intrepid2 tensor basis
  using basis_family_type = Intrepid2::HierarchicalBasisFamily<PHX::Device>;
  using tria_basis_type  = typename basis_family_type::HGRAD_TRI;
  using wedge_basis_type = typename basis_family_type::HGRAD_WEDGE;
  using wedge_basis_mono_type = Intrepid2::Basis_HGRAD_WEDGE_C1_FEM<PHX::Device,RealType,RealType>;
  using hexa_basis_type  = typename basis_family_type::HGRAD_HEX;

  auto tria_basis  = Teuchos::rcp(new tria_basis_type(1));
  auto wedge_p2_basis = Teuchos::rcp(new wedge_basis_type(2,2));
  auto wedge_mono_basis = Teuchos::rcp(new wedge_basis_mono_type());
  auto hexa_basis  = Teuchos::rcp(new hexa_basis_type(1));

  // Create field patterns for testing
  auto fp_2d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(tria_basis));
  auto fp_3d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_mono_basis));
  auto fp_3d_p2 = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_p2_basis));
  auto fp_3d_hex = Teuchos::rcp(new panzer::Intrepid2FieldPattern(hexa_basis));
  auto elem_fp = Teuchos::rcp(new panzer::ElemFieldPattern(wedge));

  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine(ts.rng_seed);
  ipdf nlay_pdf (1,5);
  ipdf ne_x_pdf (5,10);

  // Create dummy basal mesh
  GO ne_x_glb = ne_x_pdf(engine) * comm->getSize();
  int numLayers = nlay_pdf(engine);
  Teuchos::broadcast(*comm,0,1,&ne_x_glb);
  Teuchos::broadcast(*comm,0,1,&numLayers);

  // Basal mesh and monolithic (not extruded) 3d mesh
  auto mesh_2d = Teuchos::rcp(new DummyMesh(ne_x_glb,comm));

  for (auto ordering : {LayeredMeshOrdering::COLUMN,LayeredMeshOrdering::LAYER}) {
    // Build 2d and 3d conn managers
    auto conn_mgr_tria  = Teuchos::rcp(new DummyConnManager(mesh_2d));

    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set<int>("NumLayers",0);
    params->set<int>("Workset Size",1000);
    params->set("Columnwise Ordering", ordering==LayeredMeshOrdering::COLUMN);

    // Bad pointers
    TEST_THROW (Teuchos::rcp(new ExtrudedMesh(Teuchos::null,params,comm)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new ExtrudedMesh(mesh_2d,Teuchos::null,comm)),
                std::invalid_argument);

    // Bad num layers
    TEST_THROW (Teuchos::rcp(new ExtrudedMesh(mesh_2d,params,comm)),
                Teuchos::Exceptions::InvalidParameterValue);

    params->set<int>("NumLayers",numLayers);
    auto extruded_mesh = Teuchos::rcp(new ExtrudedMesh(mesh_2d,params,comm));
    extruded_mesh->setBulkData(comm);

    // Bad pointers
    TEST_THROW (Teuchos::rcp(new ExtrudedConnManager(Teuchos::null,extruded_mesh)),
                std::invalid_argument);
    TEST_THROW (Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,Teuchos::null)),
                std::invalid_argument);

    // Build extruded conn manager
    Teuchos::RCP<ConnManager> conn_mgr_ext = Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,extruded_mesh));

    // Bad field agg pattern: contains non-intrepid patterns
    auto bad_fp1 = create_agg_fp({elem_fp});
    TEST_THROW (conn_mgr_ext->buildConnectivity(bad_fp1),std::bad_cast);

    // Bad field agg pattern: contains different intrepid patterns
    auto bad_fp2 = create_agg_fp({fp_3d,fp_3d_p2});
    TEST_THROW (conn_mgr_ext->buildConnectivity(bad_fp2),std::runtime_error);

    // Bad field agg pattern: contains patterns with wrong cell topo
    auto bad_fp3 = create_agg_fp({fp_2d});
    TEST_THROW (conn_mgr_ext->buildConnectivity(bad_fp3),std::runtime_error);

    // Bad field agg pattern: basis is tens prod with wrong basal basis topology
    auto bad_fp4 = create_agg_fp({fp_3d_hex});
    TEST_THROW (conn_mgr_ext->buildConnectivity(bad_fp4),std::invalid_argument);
  }
}

TEUCHOS_UNIT_TEST(ExtrudedConnMgr, Numbering)
{
  // We test the conn generated with ExtrudedConnManager with a tensor intrepid basis
  // against the DummyConnMgr with the same tensor intrepid basis 
  using namespace Albany;

  auto comm = getDefaultComm();

  auto& ts = UnitTestSession::instance();

  // 2d, and 3d topologies

  // Create an intrepid2 tensor basis
  using basis_family_type = Intrepid2::HierarchicalBasisFamily<PHX::Device>;
  using tria_basis_type  = typename basis_family_type::HGRAD_TRI;
  using wedge_basis_type = typename basis_family_type::HGRAD_WEDGE;
  using pattern_ptr      = Teuchos::RCP<panzer::FieldPattern>;

  // Comparisong works for order_z=1, order_xy=1, but fails with either one >1.
  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine(ts.rng_seed);
  ipdf nlay_pdf (1,5);
  ipdf ne_x_pdf (1,5);
  ipdf nfields_pdf (1,3);

  int nfields = nfields_pdf(engine);
  GO ne_x_glb = ne_x_pdf(engine)*comm->getSize();
  int numElemLayers = nlay_pdf(engine);

  Teuchos::broadcast(*comm,0,1,&nfields);
  Teuchos::broadcast(*comm,0,1,&ne_x_glb);
  Teuchos::broadcast(*comm,0,1,&numElemLayers);

  // Basal mesh
  auto mesh_2d = Teuchos::rcp(new DummyMesh(ne_x_glb,comm));
  const auto num_nodes2d = mesh_2d->num_global_nodes();

  if (comm->getRank()==0) {
    std::cout << "\nRUNNING TESTS WITH:\n"
              << "  rng seed        : " << ts.rng_seed << "\n"
              << "  num glb_2d nodes: " << mesh_2d->num_global_nodes() << "\n"
              << "  num lcl_2d nodes: " << mesh_2d->get_num_local_nodes() << "\n"
              << "  num glb_2d elems: " << mesh_2d->num_global_elems() << "\n"
              << "  num lcl_2d elems: " << mesh_2d->get_num_local_elements() << "\n"
              << "  num layers      : " << numElemLayers << "\n"
              << "  nfields         : " << nfields << "\n";
  }

  for (int order_z : {1,2}) {
    if (comm->getRank()==0) {
      std::cout << " -> FE vertical order: " << order_z << "\n";
    }
    auto wedge_basis = Teuchos::rcp(new wedge_basis_type(1,order_z));

    // Create field patterns for testing
    auto fp_3d = Teuchos::rcp(new panzer::Intrepid2FieldPattern(wedge_basis));

    // Create dummy basal mesh
    auto numDofLayers = order_z*numElemLayers + 1;

    for (auto ordering : {LayeredMeshOrdering::LAYER,LayeredMeshOrdering::COLUMN} ) {

      if (comm->getRank()==0) {
        std::cout << "   -> ordering: " << (ordering==LayeredMeshOrdering::COLUMN ? "COLUMN" : "LAYER") << "\n";
      }

      // Build extruded mesh
      auto params = Teuchos::rcp(new Teuchos::ParameterList());
      params->set<int>("NumLayers",numElemLayers);
      params->set<int>("Workset Size",1000);
      params->set("Columnwise Ordering", ordering==LayeredMeshOrdering::COLUMN);
      auto extruded_mesh = Teuchos::rcp(new ExtrudedMesh(mesh_2d,params,comm));
      extruded_mesh->setBulkData(comm);

      // Build 2d conn manager
      auto conn_mgr_tria  = Teuchos::rcp(new DummyConnManager(mesh_2d));

      // Build extruded conn manager
      Teuchos::RCP<ConnManager> conn_mgr_ext = Teuchos::rcp(new ExtrudedConnManager(conn_mgr_tria,extruded_mesh));
      conn_mgr_ext->buildConnectivity(create_agg_fp(std::vector<pattern_ptr>(nfields,fp_3d)));

      // Helper structures
      const int ndofs_2d = nfields*num_nodes2d;
      auto cell_layers_lid = extruded_mesh->cell_layers_lid();
      auto dofs_layers_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(ndofs_2d,numDofLayers,ordering));

      // Check
      for (int icol=0; icol<mesh_2d->get_num_local_elements(); ++icol) {
        const auto conn2d = conn_mgr_tria->getConnectivity(icol);
        const auto size2d = conn_mgr_tria->getConnectivitySize(icol);

        for (int ilev=0; ilev<numElemLayers; ++ilev) {
          int ie = cell_layers_lid->getId(icol,ilev);
          const auto conn3d = conn_mgr_ext->getConnectivity(ie);
          const auto size3d = conn_mgr_ext->getConnectivitySize(ie);
          TEST_EQUALITY( size3d, nfields*size2d*(order_z+1) );

          // Test that 3d dofs on particular in-elem layer are the expected ones
          // Check by computing expected dof from 2d dof and the dof level
          auto test_dof_lev = [&](const int dof_lev, const int offset3d) {
            for (int idof=0; idof<size2d; ++idof) {
              for (int ifield=0; ifield<nfields; ++ifield) {
                auto dof2d = nfields*conn2d[idof] + ifield;
                auto dof3d = conn3d[offset3d+idof*nfields+ifield];
                TEST_EQUALITY (dof3d,dofs_layers_gid->getId(dof2d,dof_lev));
              }
            }
          };

          for (int ilay=0; ilay<=order_z; ++ilay) {
            int dof_lev = ilev*order_z + ilay;

            test_dof_lev(dof_lev,ilay*nfields*size2d); 
          }
        }
      }
    }
  }
}
