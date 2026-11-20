//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests for the storage of 3d output states of the logically extruded mesh
// in the basal mesh (ExtrudedMeshFieldAccessor), using an STK2D basal mesh:
//  - 3d nodal outputs get a basal (Cell2d,Node2d[,Dim],Level) layered state,
//    3d elem scalar outputs a basal (Cell2d,Layer) layered state
//  - the basal ${name}_NLC holds the mesh levels (nodal) or the layers midpoints (elem)
//  - 3d elem outputs view the basal state
//  - transferElemStateToNodeState stores in the basal state the average of the
//    layers sharing each level, and moves it into the basal nodal field
//  - extruded, interpolated and clashing nodal states have no basal output state
//  - global (scalar and vector) states view the mesh scalar/vector states

#include "Albany_UnitTestSession.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_StateInfoStruct.hpp"

#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr int num_layers = 3;
constexpr int nelem_x = 2;
constexpr int nelem_y = 2;
constexpr int ntria = nelem_x*nelem_y*2;
constexpr int neq = 2;
constexpr int vec_dim = 2;
constexpr int data_num_layers = 2;
constexpr double ztop = 1.0;   // thickness=surface_height=ztop, so the bed is at z=0
constexpr double dummy_value = 7.0;
constexpr int gvec_size = 3;
constexpr double tol = 1e-12;

// Smooth 3d function, used to fill the 3d nodal outputs (comp j of vector outputs)
double g (const double x, const double y, const double z, const int j = 0) {
  return 1 + (j+1)*x + 2*y + 10*z + j;
}

// Layer-dependent perturbation, so that the 3d elems sharing a node hold different values
constexpr double pert = 5.0;

Teuchos::RCP<Albany::AbstractDiscretization>
create_disc (const Teuchos::RCP<const Teuchos_Comm>& comm, const bool columnwise)
{
  using namespace Albany;

  auto top_level_params = Teuchos::rcp(new Teuchos::ParameterList(""));
  auto disc_params = Teuchos::sublist(top_level_params,"Discretization");
  disc_params->set<std::string>("Method", "Extruded");
  disc_params->set<int>("NumLayers", num_layers);
  disc_params->set<int>("Number Of Time Derivatives", 0);
  disc_params->set<bool>("Columnwise Ordering", columnwise);
  disc_params->set<Teuchos::Array<std::string>>("Extrude Basal Fields",{"beta"});
  disc_params->set<Teuchos::Array<std::string>>("Interpolate Basal Layered Fields",{"v"});
  disc_params->sublist("Side Set Discretizations").set("Side Sets",Teuchos::Array<std::string>{"basalside"});

  auto& basal_params = disc_params->sublist("Side Set Discretizations").sublist("basalside");
  basal_params.set<std::string>("Method", "STK2D");
  basal_params.set<int>("Number Of Time Derivatives", 0);
  basal_params.set<int>("1D Elements", nelem_x);
  basal_params.set<int>("2D Elements", nelem_y);
  basal_params.set<std::string>("Cell Topology", "Triangle");

  auto& basal_req = basal_params.sublist("Required Fields Info");
  basal_req.set<int>("Number Of Fields",5);
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
  auto& beta_req = basal_req.sublist("Field 2");
  beta_req.set<std::string>("Field Name","beta");
  beta_req.set<std::string>("Field Type","Node Scalar");
  beta_req.set<std::string>("Field Origin","File");
  beta_req.set<Teuchos::Array<std::string>>("Field Expression",{"x+y"});
  auto& v_req = basal_req.sublist("Field 3");
  v_req.set<std::string>("Field Name","v");
  v_req.set<std::string>("Field Type","Node Layered Vector");
  v_req.set<int>("Vector Dim",vec_dim);
  v_req.set<int>("Number Of Layers",data_num_layers);
  v_req.set<std::string>("Field Origin","File");
  v_req.set<std::string>("File Name","./v.ascii");
  auto& dummy_req = basal_req.sublist("Field 4");
  dummy_req.set<std::string>("Field Name","dummy");
  dummy_req.set<std::string>("Field Type","Node Scalar");
  dummy_req.set<std::string>("Field Origin","File");
  dummy_req.set<double>("Field Value",dummy_value);

  // Basal states
  const auto NDTEN = StateStruct::NodalDataToElemNode;
  std::map<std::string, Teuchos::RCP<StateInfoStruct> > ss_sis;
  auto& bsis = ss_sis["basalside"] = Teuchos::rcp(new StateInfoStruct());
  bsis->push_back(Teuchos::rcp(new StateStruct("thickness",NDTEN,{ntria,3})));
  bsis->push_back(Teuchos::rcp(new StateStruct("surface_height",NDTEN,{ntria,3})));
  bsis->push_back(Teuchos::rcp(new StateStruct("beta",NDTEN,{ntria,3})));
  bsis->push_back(Teuchos::rcp(new StateStruct("dummy",NDTEN,{ntria,3})));
  auto bst_v = Teuchos::rcp(new StateStruct("v",NDTEN,{ntria,3,vec_dim,data_num_layers}));
  bst_v->layered = true;
  bsis->push_back(bst_v);

  // Volume states:
  //  - beta, v: inputs (extruded/interpolated)
  //  - out_s, out_v: nodal scalar/vector outputs
  //  - out_e: elem scalar output
  //  - dummy: neither extruded nor interpolated, but clashing with a basal state
  //  - gscl, gvec: global scalar/vector states
  constexpr int nelem3d = num_layers*ntria;
  auto sis = Teuchos::rcp(new StateInfoStruct());
  sis->push_back(Teuchos::rcp(new StateStruct("beta",NDTEN,{nelem3d,6})));
  sis->push_back(Teuchos::rcp(new StateStruct("v",NDTEN,{nelem3d,6,vec_dim})));
  sis->push_back(Teuchos::rcp(new StateStruct("out_s",NDTEN,{nelem3d,6})));
  sis->push_back(Teuchos::rcp(new StateStruct("out_v",NDTEN,{nelem3d,6,vec_dim})));
  sis->push_back(Teuchos::rcp(new StateStruct("out_e",StateStruct::ElemData,{nelem3d})));
  sis->push_back(Teuchos::rcp(new StateStruct("dummy",NDTEN,{nelem3d,6})));
  sis->push_back(Teuchos::rcp(new StateStruct("gscl",StateStruct::WorksetValue,{1})));
  sis->push_back(Teuchos::rcp(new StateStruct("gvec",StateStruct::WorksetValue,{gvec_size})));

  DiscretizationFactory factory(top_level_params,comm,false);
  factory.createMeshSpecs();
  return factory.createDiscretization(neq,{},sis,ss_sis);
}

} // anonymous namespace

TEUCHOS_UNIT_TEST (ExtrudedMeshFieldAccessor, BasalOutputStates)
{
  using namespace Albany;

  auto comm = getDefaultComm();
  for (bool columnwise : {true, false}) {
    out << " -> Columnwise Ordering: " << (columnwise ? "true" : "false") << "\n";
    auto disc  = create_disc(comm,columnwise);
    auto bdisc = disc->getSideSetDiscretizations().at("basalside");
    auto mfa   = disc->getMeshStruct()->get_field_accessor();
    auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
    const auto& bsis = bmfa->getAllSIS();

    // Nodal outputs: (Cell2d,Node2d[,Dim],Level), layered.
    // NOTE: we skip dim[0], which is just a placeholder for the workset size.
    using FieldDims = StateStruct::FieldDims;
    auto dims = [&](const std::string& name) {
      const auto& d = bsis.find(name)->dim;
      return FieldDims(d.begin()+1,d.end());
    };
    const FieldDims out_s_dims = {3,num_layers+1};
    const FieldDims out_v_dims = {3,vec_dim,num_layers+1};
    TEST_ASSERT (bsis.find("out_s")->entity==StateStruct::NodalDataToElemNode);
    TEST_ASSERT (bsis.find("out_v")->entity==StateStruct::NodalDataToElemNode);
    TEST_COMPARE_ARRAYS (dims("out_s"),out_s_dims);
    TEST_COMPARE_ARRAYS (dims("out_v"),out_v_dims);
    TEST_ASSERT (bsis.find("out_s")->layered);
    TEST_ASSERT (bsis.find("out_v")->layered);

    // Elem scalar output: (Cell2d,Layer), layered
    const FieldDims out_e_dims = {num_layers};
    TEST_ASSERT (bsis.find("out_e")->entity==StateStruct::ElemData);
    TEST_COMPARE_ARRAYS (dims("out_e"),out_e_dims);
    TEST_ASSERT (bsis.find("out_e")->layered);

    // Inputs and clashing states keep the original basal state
    const FieldDims scalar_dims = {3};
    TEST_COMPARE_ARRAYS (dims("beta"),scalar_dims);
    TEST_COMPARE_ARRAYS (dims("dummy"),scalar_dims);
    TEST_ASSERT (not bsis.find("dummy")->layered);

    // Normalized layers coords: mesh levels for nodal outputs, layers midpoints for elem outputs
    const auto& z_ref = mfa->getMeshVectorStates().at("layers_z_ref");
    TEST_EQUALITY (static_cast<int>(z_ref.size()),num_layers+1);
    auto& bvec_states = bmfa->getMeshVectorStates();
    TEST_COMPARE_FLOATING_ARRAYS (bvec_states.at("out_s_NLC"),z_ref,tol);
    TEST_COMPARE_FLOATING_ARRAYS (bvec_states.at("out_v_NLC"),z_ref,tol);
    std::vector<double> mid(num_layers);
    for (int il=0; il<num_layers; ++il) {
      mid[il] = (z_ref[il]+z_ref[il+1])/2;
    }
    TEST_COMPARE_FLOATING_ARRAYS (bvec_states.at("out_e_NLC"),mid,tol);
    TEST_ASSERT (bvec_states.count("dummy_NLC")==0);
    TEST_ASSERT (bvec_states.count("beta_NLC")==0);
  }
}

TEUCHOS_UNIT_TEST (ExtrudedMeshFieldAccessor, ElemOutputs)
{
  using namespace Albany;

  auto comm = getDefaultComm();
  for (bool columnwise : {true, false}) {
    out << " -> Columnwise Ordering: " << (columnwise ? "true" : "false") << "\n";
    auto disc  = create_disc(comm,columnwise);
    auto bdisc = disc->getSideSetDiscretizations().at("basalside");
    auto mfa   = disc->getMeshStruct()->get_field_accessor();
    auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
    const auto& coords  = disc->getCoords();
    const auto& bcoords = bdisc->getCoords();
    const auto& mid = bmfa->getMeshVectorStates().at("out_e_NLC");

    // Fill the 3d elem output with a function of the elem centroid. Since the 3d state
    // views the basal one, the basal state must hold the value of the 3d elem at
    // (Cell2d,Layer), whose centroid is (basal elem centroid, layer midpoint).
    const int num_ws = disc->getNumWorksets();
    TEST_EQUALITY (num_ws,bdisc->getNumWorksets());
    for (int ws=0; ws<num_ws; ++ws) {
      auto& state = mfa->getElemStates()[ws].at("out_e");
      auto state_h = state.host();
      const int nelem3d = state_h.extent_int(0);
      for (int ie=0; ie<nelem3d; ++ie) {
        const int nn = coords[ws][ie].size();
        double c[3] = {0,0,0};
        for (int in=0; in<nn; ++in) {
          for (int d=0; d<3; ++d) {
            c[d] += coords[ws][ie][in][d] / nn;
          }
        }
        state_h(ie) = g(c[0],c[1],c[2]);
      }
      state.sync_to_dev();
    }

    for (int ws=0; ws<num_ws; ++ws) {
      auto bstate_h = bmfa->getElemStates()[ws].at("out_e").host();
      const int nbelem = bstate_h.extent_int(0);
      TEST_EQUALITY (bstate_h.extent_int(1),num_layers);
      for (int ie=0; ie<nbelem; ++ie) {
        const int nn = bcoords[ws][ie].size();
        double c[2] = {0,0};
        for (int in=0; in<nn; ++in) {
          for (int d=0; d<2; ++d) {
            c[d] += bcoords[ws][ie][in][d] / nn;
          }
        }
        for (int il=0; il<num_layers; ++il) {
          TEST_FLOATING_EQUALITY (bstate_h(ie,il),g(c[0],c[1],ztop*mid[il]),tol);
        }
      }
    }

    // Not a nodal state
    TEST_THROW (mfa->transferElemStateToNodeState("out_e"),std::runtime_error);
  }
}

TEUCHOS_UNIT_TEST (ExtrudedMeshFieldAccessor, NodalOutputs)
{
  using namespace Albany;

  auto comm = getDefaultComm();
  for (bool columnwise : {true, false}) {
    out << " -> Columnwise Ordering: " << (columnwise ? "true" : "false") << "\n";
    auto disc  = create_disc(comm,columnwise);
    auto bdisc = disc->getSideSetDiscretizations().at("basalside");
    auto mfa   = disc->getMeshStruct()->get_field_accessor();
    auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
    const auto& coords  = disc->getCoords();
    const auto& bcoords = bdisc->getCoords();
    const auto& z_ref = mfa->getMeshVectorStates().at("layers_z_ref");
    const int num_ws = disc->getNumWorksets();

    // Average of the centroid z of the 3d elems containing a node at level lev
    std::vector<double> zc_avg(num_layers+1);
    for (int lev=0; lev<=num_layers; ++lev) {
      const double zc_below = lev>0          ? (z_ref[lev-1]+z_ref[lev])/2 : 0;
      const double zc_above = lev<num_layers ? (z_ref[lev]+z_ref[lev+1])/2 : 0;
      const int    nadj     = (lev>0 ? 1 : 0) + (lev<num_layers ? 1 : 0);
      zc_avg[lev] = ztop*(zc_below+zc_above)/nadj;
    }

    // Fill the 3d nodal outputs (and the clashing state) as (Cell,Node[,Dim]) elem states:
    // a smooth nodal function, plus a perturbation proportional to the elem centroid z,
    // so that the 3d elems sharing a node hold different values.
    for (int ws=0; ws<num_ws; ++ws) {
      auto& s = mfa->getElemStates()[ws].at("out_s");
      auto& v = mfa->getElemStates()[ws].at("out_v");
      auto& d = mfa->getElemStates()[ws].at("dummy");
      auto s_h = s.host();
      auto v_h = v.host();
      auto d_h = d.host();
      const int nelem3d = s_h.extent_int(0);
      for (int ie=0; ie<nelem3d; ++ie) {
        const int nn = coords[ws][ie].size();
        double zc = 0;
        for (int in=0; in<nn; ++in) {
          zc += coords[ws][ie][in][2] / nn;
        }
        for (int in=0; in<nn; ++in) {
          const double* xyz = coords[ws][ie][in];
          s_h(ie,in) = g(xyz[0],xyz[1],xyz[2]) + pert*zc;
          for (int j=0; j<vec_dim; ++j) {
            v_h(ie,in,j) = g(xyz[0],xyz[1],xyz[2],j) + pert*zc;
          }
          d_h(ie,in) = -1;
        }
      }
      s.sync_to_dev();
      v.sync_to_dev();
      d.sync_to_dev();
    }

    // Expected basal (Cell2d,Node2d[,Dim],Level) values
    auto check_basal = [&](const std::string& msg) {
      out << "    checking basal states " << msg << "\n";
      for (int ws=0; ws<num_ws; ++ws) {
        auto s_h = bmfa->getElemStates()[ws].at("out_s").host();
        auto v_h = bmfa->getElemStates()[ws].at("out_v").host();
        const int nbelem = s_h.extent_int(0);
        for (int ie=0; ie<nbelem; ++ie) {
          const int nn = bcoords[ws][ie].size();
          for (int in=0; in<nn; ++in) {
            const double* xy = bcoords[ws][ie][in];
            for (int lev=0; lev<=num_layers; ++lev) {
              const double z = ztop*z_ref[lev];
              TEST_FLOATING_EQUALITY (s_h(ie,in,lev),g(xy[0],xy[1],z)+pert*zc_avg[lev],tol);
              for (int j=0; j<vec_dim; ++j) {
                TEST_FLOATING_EQUALITY (v_h(ie,in,j,lev),g(xy[0],xy[1],z,j)+pert*zc_avg[lev],tol);
              }
            }
          }
        }
      }
    };

    // All ranks must call the transfer, since the basal mesh sums over ranks
    mfa->transferElemStateToNodeState("out_s");
    mfa->transferElemStateToNodeState("out_v");
    check_basal("after transferElemStateToNodeState");

    // The basal nodal fields must hold the same values: zero the basal elem states,
    // and load them back from the nodal fields
    for (int ws=0; ws<num_ws; ++ws) {
      for (const std::string name : {"out_s","out_v"}) {
        auto& bstate = bmfa->getElemStates()[ws].at(name);
        Kokkos::deep_copy(bstate.host(),0.0);
        bstate.sync_to_dev();
      }
    }
    bmfa->transferNodeStatesToElemStates();
    check_basal("loaded from the basal nodal fields");

    // Extruded, interpolated and clashing states: no-op (the basal states keep their values)
    mfa->transferElemStateToNodeState("beta");
    mfa->transferElemStateToNodeState("v");
    mfa->transferElemStateToNodeState("dummy");
    bmfa->transferNodeStatesToElemStates();
    for (int ws=0; ws<num_ws; ++ws) {
      auto d_h = bmfa->getElemStates()[ws].at("dummy").host();
      auto b_h = bmfa->getElemStates()[ws].at("beta").host();
      const int nbelem = d_h.extent_int(0);
      for (int ie=0; ie<nbelem; ++ie) {
        for (int in=0; in<d_h.extent_int(1); ++in) {
          const double* xy = bcoords[ws][ie][in];
          TEST_FLOATING_EQUALITY (d_h(ie,in),dummy_value,tol);
          TEST_FLOATING_EQUALITY (b_h(ie,in)+1,xy[0]+xy[1]+1,tol); // beta = x+y, shifted away from 0
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST (ExtrudedMeshFieldAccessor, GlobalStates)
{
  using namespace Albany;

  auto comm = getDefaultComm();
  auto disc = create_disc(comm,true);
  auto mfa  = disc->getMeshStruct()->get_field_accessor();

  // Global states have rank 1: scalars if dim[0]==1, vectors of length dim[0] otherwise.
  // Their host views alias the mesh scalar/vector states.
  auto& gstates = mfa->getGlobalStates();
  const auto& gscl_h = gstates.at("gscl").host();
  const auto& gvec_h = gstates.at("gvec").host();
  TEST_EQUALITY (gscl_h.extent_int(0),1);
  TEST_EQUALITY (gvec_h.extent_int(0),gvec_size);

  gscl_h(0) = 2.0;
  for (int i=0; i<gvec_size; ++i) {
    gvec_h(i) = i+1;
  }

  TEST_EQUALITY (mfa->getMeshScalarStates().at("gscl"),2.0);
  const auto& gvec = mfa->getMeshVectorStates().at("gvec");
  TEST_EQUALITY (static_cast<int>(gvec.size()),gvec_size);
  for (int i=0; i<gvec_size; ++i) {
    TEST_EQUALITY (gvec[i],i+1.0);
  }
}
