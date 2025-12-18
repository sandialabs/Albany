//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_UnitTestSession.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_DiscretizationFactory.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <random>

constexpr int num_layers = 3;
constexpr int nelem_x = 2;
constexpr int nelem_y = 2;
constexpr int ntria = nelem_x*nelem_y*2;
constexpr int neq = 2;
constexpr int data_num_layers = 2;
constexpr double ztop = 1.0;

Teuchos::RCP<Albany::AbstractDiscretization>
create_extruded_disc (const Teuchos::RCP<const Teuchos_Comm>& comm, bool logically_extruded)
{
  using namespace Albany;

  auto top_level_params = Teuchos::rcp(new Teuchos::ParameterList(""));
  auto disc_params = Teuchos::sublist(top_level_params,"Discretization");
  disc_params->set<int>("NumLayers", num_layers);
  disc_params->set<int>("Number Of Time Derivatives", 0);
  disc_params->set<bool>("Columnwise Ordering", true);
  disc_params->sublist("Side Set Discretizations").set("Side Sets",Teuchos::Array<std::string>{"basalside"});
  auto& basal_params = disc_params->sublist("Side Set Discretizations").sublist("basalside");
  basal_params.set<std::string>("Method", "STK2D");
  basal_params.set<int>("Number Of Time Derivatives", 0);
  basal_params.set<int>("1D Elements", nelem_x);
  basal_params.set<int>("2D Elements", nelem_y);
  basal_params.set<std::string>("Cell Topology", "Triangle");

  auto& basal_req = basal_params.sublist("Required Fields Info");
  basal_req.set<int>("Number Of Fields",4);
  auto& H_req = basal_req.sublist("Field 0");
  H_req.set<std::string>("Field Name","thickness");
  H_req.set<std::string>("Field Type","Node Scalar");
  H_req.set<std::string>("Field Origin","File");
  H_req.set<double>("Field Value",ztop);
  auto& zs_req = basal_req.sublist("Field 1");
  zs_req.set<std::string>("Field Name","surface_height");
  zs_req.set<std::string>("Field Type","Node Scalar");
  zs_req.set<std::string>("Field Origin","File");
  zs_req.set<double>("Field Value",ztop);
  auto& scl_req = basal_req.sublist("Field 2");
  scl_req.set<std::string>("Field Name","beta");
  scl_req.set<std::string>("Field Type","Node Scalar");
  scl_req.set<std::string>("Field Origin","File");
  scl_req.set<Teuchos::Array<std::string>>("Field Expression",{"x+y"});
  auto& vec_lay_req = basal_req.sublist("Field 3");
  vec_lay_req.set<std::string>("Field Name","v");
  vec_lay_req.set<std::string>("Field Type","Node Layered Vector");
  vec_lay_req.set<int>("Vector Dim",2);
  vec_lay_req.set<int>("Number Of Layers",data_num_layers);
  vec_lay_req.set<std::string>("Field Origin","File");
  vec_lay_req.set<std::string>("File Name","./v.ascii");

  // Basal state info structs
  auto bst_H    = Teuchos::rcp(new StateStruct("thickness",StateStruct::NodalDataToElemNode,{ntria,3}));
  auto bst_zs   = Teuchos::rcp(new StateStruct("surface_height",StateStruct::NodalDataToElemNode,{ntria,3}));
  auto bst_beta = Teuchos::rcp(new StateStruct("beta",StateStruct::NodalDataToElemNode,{ntria,3}));
  auto bst_v    = Teuchos::rcp(new StateStruct("v",StateStruct::NodalDataToElemNode,{ntria,3,neq,data_num_layers}));
  bst_v->layered = true;

  std::map<std::string, Teuchos::RCP<StateInfoStruct> > ss_sis;
  auto& bsis = ss_sis["basalside"] = Teuchos::rcp(new StateInfoStruct());
  bsis->push_back(bst_H);
  bsis->push_back(bst_zs);
  bsis->push_back(bst_beta);
  bsis->push_back(bst_v);

  if (logically_extruded) {
    disc_params->set<std::string>("Method", "Extruded");
    disc_params->set<Teuchos::Array<std::string>>("Extrude Basal Fields",{"beta"});
    disc_params->set<Teuchos::Array<std::string>>("Interpolate Basal Layered Fields",{"v"});
  } else {
    disc_params->set<std::string>("Method", "STKExtruded");
    disc_params->set<Teuchos::Array<std::string>>("Extrude Basal Node Fields",{"beta"});
    disc_params->set<Teuchos::Array<int>>("Basal Node Fields Ranks",{1});
    disc_params->set<Teuchos::Array<std::string>>("Interpolate Basal Node Layered Fields",{"v"});
    disc_params->set<Teuchos::Array<int>>("Basal Node Layered Fields Ranks",{2});

    auto& req = disc_params->sublist("Required Fields Info");
    req.set<int>("Number Of Fields",2);
    auto& beta_req = req.sublist("Field 0");
    beta_req.set<std::string>("Field Name","beta");
    beta_req.set<std::string>("Field Type","Node Scalar");
    beta_req.set<std::string>("Field Origin","Mesh");
    auto& v_req = req.sublist("Field 1");
    v_req.set<std::string>("Field Name","v");
    v_req.set<std::string>("Field Type","Node Vector");
    v_req.set<int>("Vector Dim",2);
    v_req.set<std::string>("Field Origin","Mesh");
  }

  // Volume state info structs
  auto st_beta = Teuchos::rcp(new StateStruct("beta",StateStruct::NodalDataToElemNode,{num_layers*ntria,6}));
  auto st_v = Teuchos::rcp(new StateStruct("v",StateStruct::NodalDataToElemNode,{num_layers*ntria,6,neq}));

  auto sis = Teuchos::rcp(new StateInfoStruct());
  sis->push_back(st_beta);
  sis->push_back(st_v);

  DiscretizationFactory factory(top_level_params,comm,false);

  factory.createMeshSpecs();
  auto disc = factory.createDiscretization(neq,{},sis,ss_sis);

  return disc;
}

TEUCHOS_UNIT_TEST (ExtrudedDisc,Coordinates)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto disc = create_extruded_disc(comm,true);
  auto stk_disc = create_extruded_disc(comm,false);
  auto bdisc = disc->getSideSetDiscretizations().at("basalside");

  // +-------------------------------------------------------+
  // |       Test coordinates against manual extrusion       |
  // +-------------------------------------------------------+

  const auto& coords = disc->getCoords();
  const auto& bcoords = bdisc->getCoords();
  const auto& layers_data = disc->getMeshStruct()->layers_data;

  TEST_EQUALITY(layers_data.cell.gid->numLayers,num_layers);
  TEST_EQUALITY(layers_data.cell.lid->numLayers,num_layers);

  TEST_EQUALITY(disc->getNumWorksets(),bdisc->getNumWorksets());
  int num_ws = disc->getNumWorksets();
  auto ws_sizes = disc->getWorksetsSizes();
  auto bws_sizes = bdisc->getWorksetsSizes();

  auto ws_elems = disc->getWsElementLIDs().host();
  auto bws_elems = bdisc->getWsElementLIDs().host();

  TEST_EQUALITY (coords.size(),num_ws);
  TEST_EQUALITY (bcoords.size(),num_ws);

  int num_belem_nodes = bcoords[0][0].size();
  double dz = ztop / num_layers;
  for (int ws=0; ws<coords.size(); ++ws) {
    TEST_EQUALITY(coords[ws].size(),bcoords[ws].size()*num_layers);
    for (int ibe=0; ibe<bcoords[ws].size(); ++ibe) {
      for (int in=0; in<num_belem_nodes; ++in) {
        auto bxy = bcoords[ws][ibe][in];
        for (int il=0; il<num_layers; ++il) {
          int ie = layers_data.cell.lid->getId(ibe,il);
          for (int lev : {0,1}) {
            auto xyz = coords[ws][ie][in+lev*num_belem_nodes];
            TEST_EQUALITY (bxy[0],xyz[0]);
            TEST_EQUALITY (bxy[1],xyz[1]);

            TEST_FLOATING_EQUALITY (xyz[2],dz * (il+lev),1e-7);
          }
        }
      }
    }
  }

  // +-------------------------------------------------------+
  // |       Test coordinates against stk discretization     |
  // +-------------------------------------------------------+

  const auto& stk_coords = stk_disc->getCoords();

  TEST_EQUALITY(disc->getNumWorksets(),stk_disc->getNumWorksets());

  TEST_EQUALITY (stk_coords.size(),num_ws);

  for (int ws=0; ws<coords.size(); ++ws) {
    TEST_EQUALITY(coords[ws].size(),stk_coords[ws].size());
    for (int ie=0; ie<coords[ws].size(); ++ie) {
      TEST_EQUALITY(coords[ws][ie].size(),stk_coords[ws][ie].size());
      for (int in=0; in<coords[ws][ie].size(); ++in) {
        auto xyz = coords[ws][ie][in];
        auto stk_xyz = stk_coords[ws][ie][in];
        TEST_EQUALITY (xyz[0],stk_xyz[0]);
        TEST_EQUALITY (xyz[1],stk_xyz[1]);
        TEST_FLOATING_EQUALITY (xyz[2],stk_xyz[2],1e-7);
      }
    }
  }
}

TEUCHOS_UNIT_TEST (ExtrudedDisc,DOFs)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto disc = create_extruded_disc(comm,true);
  auto stk_disc = create_extruded_disc(comm,false);

  // +-------------------------------------------------------+
  // |        Test dofs ids against stk discretization       |
  // +-------------------------------------------------------+

  const auto& dof_mgr = disc->getDOFManager();
  const auto& dof_mgr_stk = disc->getDOFManager();

  TEST_EQUALITY(dof_mgr->cell_indexer()->getNumLocalElements(),
                dof_mgr_stk->cell_indexer()->getNumLocalElements());

  const int num_elems = dof_mgr->cell_indexer()->getNumLocalElements();
  const auto& lids = dof_mgr->elem_dof_lids().host();
  const auto& lids_stk = dof_mgr->elem_dof_lids().host();
  TEST_EQUALITY(lids.extent(1),lids_stk.extent(1));
  const int ndofs = lids.extent(1);
  for (int ie=0; ie<num_elems; ++ie) {
    const auto& gids = dof_mgr->getElementGIDs(ie);
    const auto& gids_stk = dof_mgr_stk->getElementGIDs(ie);

    TEST_COMPARE_ARRAYS(gids,gids_stk);

    for (int idof=0; idof<ndofs; ++idof) {
      TEST_EQUALITY(lids(ie,idof),lids_stk(ie,idof));
    }
  }
}

TEUCHOS_UNIT_TEST (ExtrudedDisc,LoadFields)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto disc = create_extruded_disc(comm,true);
  auto stk_disc = create_extruded_disc(comm,false);

  // +-------------------------------------------------------+
  // |        Test extruded and interpolated fields          |
  // +-------------------------------------------------------+

  auto mfa = disc->getMeshStruct()->get_field_accessor();
  auto mfa_stk = stk_disc->getMeshStruct()->get_field_accessor();

  const int num_ws = disc->getNumWorksets();
  for (int ws=0; ws<num_ws; ++ws) {
    auto beta = mfa->getElemStates()[ws].at("beta").host();
    auto beta_stk = mfa_stk->getElemStates()[ws].at("beta").host();
    TEST_EQUALITY (beta.extent(0),beta_stk.extent(0));
    TEST_EQUALITY (beta.extent(1),beta_stk.extent(1));
    int nelems = beta.extent(0);
    int nnodes = beta.extent(1);
    for (int ie=0; ie<nelems; ++ie) {
      for (int in=0; in<nnodes; ++in) {
        TEST_EQUALITY(beta(ie,in),beta_stk(ie,in));
      }
    }

    auto v = mfa->getElemStates()[ws].at("v").host();
    auto v_stk = mfa_stk->getElemStates()[ws].at("v").host();
    TEST_EQUALITY (v.extent(0),v_stk.extent(0));
    TEST_EQUALITY (v.extent(1),v_stk.extent(1));
    TEST_EQUALITY (v.extent(2),v_stk.extent(2));

    for (int ie=0; ie<nelems; ++ie) {
      for (int in=0; in<nnodes; ++in) {
        for (int eq=0; eq<neq; ++eq) {
          TEST_EQUALITY(v(ie,in,eq),v_stk(ie,in,eq));
        }
      }
    }
  }
}
