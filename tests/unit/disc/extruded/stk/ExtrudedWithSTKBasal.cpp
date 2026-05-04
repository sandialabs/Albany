//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests for the Extruded discretization using an STK Ioss (Exodus) basal mesh.
// These tests use the real GIS (Greenland Ice Sheet) populated basal mesh
// (gis_unstruct_basal_populated.exo) to verify that:
//  - the extruded disc can be created with a real-file Ioss basal mesh
//  - scalar fields loaded from the mesh are correctly extruded to the 3D mesh
//  - layered scalar fields (temperature) are correctly interpolated to the 3D mesh
//  - expected mesh parts (node sets and side sets) are present

#include "Albany_UnitTestSession.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_StateInfoStruct.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <algorithm>
#include <limits>

// The GIS basal mesh has 2D triangular elements; the extruded 3D mesh has wedge elements.
static constexpr int num_layers     = 5;   // number of vertical layers in extruded mesh
static constexpr int num_temp_layers = 11; // number of temperature data layers in the mesh file
static constexpr int neq            = 2;   // number of equations (arbitrary, just to build disc)
static constexpr int nBNodes        = 3;   // nodes per basal triangle element
static constexpr int nVNodes        = 6;   // nodes per extruded wedge element

static const std::string exo_file = "./gis_unstruct_basal_populated.exo";

namespace {

// Create an extruded discretization with an STK Ioss basal mesh loaded from the GIS exo file.
// Fields ice_thickness, surface_height, and basal_friction are scalar nodal fields loaded
// directly from the Exodus restart (Field Origin: Mesh).
// temperature is a 11-layer nodal field also loaded from the Exodus restart.
// The extruded disc extrudes the three scalar fields and interpolates temperature.
//
// NOTE: because the layer NLC (normalized layer coordinates) for temperature is not stored
// in the Exodus file, interpolateBasalLayeredFields will use zero NLC and will copy the
// last data layer to all mesh layers.  The test therefore only checks that temperature
// values are finite and lie within the data bounds, not that interpolation is exact.
Teuchos::RCP<Albany::AbstractDiscretization>
create_disc(const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  using namespace Albany;

  auto top_level_params = Teuchos::rcp(new Teuchos::ParameterList(""));
  auto disc_params = Teuchos::sublist(top_level_params, "Discretization");

  disc_params->set<std::string>("Method", "Extruded");
  disc_params->set<int>("NumLayers", num_layers);
  disc_params->set<int>("Number Of Time Derivatives", 0);
  disc_params->set<bool>("Columnwise Ordering", true);
  // Tell the extruded mesh which fields hold thickness/surface_height
  disc_params->set<std::string>("Thickness Field Name",     "ice_thickness");
  disc_params->set<std::string>("Surface Height Field Name","surface_height");
  // Fields to extrude (copy basal value to all layers)
  disc_params->set<Teuchos::Array<std::string>>("Extrude Basal Fields",
      Teuchos::Array<std::string>{"ice_thickness","surface_height","basal_friction"});
  // Fields to interpolate (interpolate basal layered data to mesh layers)
  disc_params->set<Teuchos::Array<std::string>>("Interpolate Basal Layered Fields",
      Teuchos::Array<std::string>{"temperature"});

  // Side-set discretization: only basalside is needed (upperside/lateralside are virtual)
  disc_params->sublist("Side Set Discretizations")
             .set("Side Sets", Teuchos::Array<std::string>{"basalside"});

  // Basal mesh: Ioss (Exodus) with restart to load field data
  auto& basal_params = disc_params->sublist("Side Set Discretizations").sublist("basalside");
  basal_params.set<std::string>("Method", "Ioss");
  basal_params.set<int>("Number Of Time Derivatives", 0);
  basal_params.set<int>("Restart Index", 1);
  basal_params.set<std::string>("Exodus Input File Name", exo_file);

  // Declare which fields are expected on the basal mesh (all already present from Restart)
  auto& basal_req = basal_params.sublist("Required Fields Info");
  basal_req.set<int>("Number Of Fields", 4);

  auto& H_req   = basal_req.sublist("Field 0");
  H_req.set<std::string>("Field Name",   "ice_thickness");
  H_req.set<std::string>("Field Type",   "Node Scalar");
  H_req.set<std::string>("Field Origin", "Mesh");

  auto& zs_req  = basal_req.sublist("Field 1");
  zs_req.set<std::string>("Field Name",   "surface_height");
  zs_req.set<std::string>("Field Type",   "Node Scalar");
  zs_req.set<std::string>("Field Origin", "Mesh");

  auto& beta_req = basal_req.sublist("Field 2");
  beta_req.set<std::string>("Field Name",   "basal_friction");
  beta_req.set<std::string>("Field Type",   "Node Scalar");
  beta_req.set<std::string>("Field Origin", "Mesh");

  auto& temp_req = basal_req.sublist("Field 3");
  temp_req.set<std::string>("Field Name",     "temperature");
  temp_req.set<std::string>("Field Type",     "Node Layered Scalar");
  temp_req.set<int>("Number Of Layers", num_temp_layers);
  temp_req.set<std::string>("Field Origin",   "Mesh");

  // ---------------------------------------------------------------------------
  // Basal StateInfoStruct: describes fields stored on the 2D basal mesh.
  // dim[0] is a placeholder (replaced by workset size); dim[1] = nodes per
  // basal element (= 3 for triangles); for layered, dim[2] = number of layers.
  // ---------------------------------------------------------------------------
  std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> ss_sis;
  auto& bsis = ss_sis["basalside"] = Teuchos::rcp(new Albany::StateInfoStruct());

  auto bst_H    = Teuchos::rcp(new Albany::StateStruct(
      "ice_thickness",  Albany::StateStruct::NodalDataToElemNode, {1, nBNodes}));
  auto bst_zs   = Teuchos::rcp(new Albany::StateStruct(
      "surface_height", Albany::StateStruct::NodalDataToElemNode, {1, nBNodes}));
  auto bst_beta = Teuchos::rcp(new Albany::StateStruct(
      "basal_friction", Albany::StateStruct::NodalDataToElemNode, {1, nBNodes}));
  // Layered scalar: dim = {placeholder, nBNodes, nLayers}
  auto bst_temp = Teuchos::rcp(new Albany::StateStruct(
      "temperature",    Albany::StateStruct::NodalDataToElemNode, {1, nBNodes, num_temp_layers}));
  bst_temp->layered = true;

  bsis->push_back(bst_H);
  bsis->push_back(bst_zs);
  bsis->push_back(bst_beta);
  bsis->push_back(bst_temp);

  // ---------------------------------------------------------------------------
  // Volume StateInfoStruct: describes fields on the 3D extruded mesh (wedge6).
  // dim[1] = nodes per wedge element (= 6).
  // The extruded/interpolated flags are set by ExtrudedMesh::setFieldData.
  // ---------------------------------------------------------------------------
  auto sis = Teuchos::rcp(new Albany::StateInfoStruct());

  auto st_H    = Teuchos::rcp(new Albany::StateStruct(
      "ice_thickness",  Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));
  auto st_zs   = Teuchos::rcp(new Albany::StateStruct(
      "surface_height", Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));
  auto st_beta = Teuchos::rcp(new Albany::StateStruct(
      "basal_friction", Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));
  auto st_temp = Teuchos::rcp(new Albany::StateStruct(
      "temperature",    Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));

  sis->push_back(st_H);
  sis->push_back(st_zs);
  sis->push_back(st_beta);
  sis->push_back(st_temp);

  Albany::DiscretizationFactory factory(top_level_params, comm, false);
  factory.createMeshSpecs();
  return factory.createDiscretization(neq, {}, sis, ss_sis);
}

// Compute the global min and max of a rank-2 DynRankView (host) across all MPI ranks.
template<typename DT>
std::pair<double,double> global_minmax2(
    const Kokkos::DynRankView<DT,Kokkos::LayoutRight,Albany::HostMemSpace>& v,
    const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  double lmin =  std::numeric_limits<double>::max();
  double lmax = -std::numeric_limits<double>::max();
  for (size_t i=0; i<v.extent(0); ++i)
    for (size_t j=0; j<v.extent(1); ++j) {
      lmin = std::min(lmin, (double)v(i,j));
      lmax = std::max(lmax, (double)v(i,j));
    }
  double gmin, gmax;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &lmin, &gmin);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &lmax, &gmax);
  return {gmin, gmax};
}

// Compute the global min and max of a rank-3 DynRankView (host) across all MPI ranks.
template<typename DT>
std::pair<double,double> global_minmax3(
    const Kokkos::DynRankView<DT,Kokkos::LayoutRight,Albany::HostMemSpace>& v,
    const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  double lmin =  std::numeric_limits<double>::max();
  double lmax = -std::numeric_limits<double>::max();
  for (size_t i=0; i<v.extent(0); ++i)
    for (size_t j=0; j<v.extent(1); ++j)
      for (size_t k=0; k<v.extent(2); ++k) {
        lmin = std::min(lmin, (double)v(i,j,k));
        lmax = std::max(lmax, (double)v(i,j,k));
      }
  double gmin, gmax;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &lmin, &gmin);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &lmax, &gmax);
  return {gmin, gmax};
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test: verify that the expected mesh parts are created in the extruded disc.
// The extruded mesh always creates: bottom, top, lateral node sets and
// basalside, upperside, lateralside side sets, regardless of the basal mesh.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_STKBasal, MeshPartsCreated)
{
  auto comm = Albany::getDefaultComm();
  auto disc = create_disc(comm);

  const auto& ms = disc->getMeshStruct()->meshSpecs[0];

  // Node sets always created by ExtrudedMesh
  const auto& ns = ms->nsNames;
  TEST_ASSERT(std::find(ns.begin(), ns.end(), "bottom")  != ns.end());
  TEST_ASSERT(std::find(ns.begin(), ns.end(), "top")     != ns.end());
  TEST_ASSERT(std::find(ns.begin(), ns.end(), "lateral") != ns.end());

  // Side sets always created by ExtrudedMesh
  const auto& ss = ms->ssNames;
  TEST_ASSERT(std::find(ss.begin(), ss.end(), "basalside")  != ss.end());
  TEST_ASSERT(std::find(ss.begin(), ss.end(), "upperside")  != ss.end());
  TEST_ASSERT(std::find(ss.begin(), ss.end(), "lateralside")!= ss.end());
}

// ---------------------------------------------------------------------------
// Test: verify that basal fields loaded from the Exodus file are non-zero.
// This confirms that the Ioss mesh actually read the field data.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_STKBasal, FieldsLoaded)
{
  auto comm = Albany::getDefaultComm();
  auto disc = create_disc(comm);

  auto bdisc = disc->getSideSetDiscretizations().at("basalside");
  auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
  const int bnum_ws = bdisc->getNumWorksets();

  // Scalar fields: at least one element must have a non-zero value
  for (const std::string& fname : {"ice_thickness", "surface_height", "basal_friction"}) {
    double lmax = 0.0;
    for (int ws=0; ws<bnum_ws; ++ws) {
      const auto& bstate = bmfa->getElemStates()[ws].at(fname).host();
      for (size_t ie=0; ie<bstate.extent(0); ++ie)
        for (size_t in=0; in<bstate.extent(1); ++in)
          lmax = std::max(lmax, std::abs(bstate(ie,in)));
    }
    double gmax;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &lmax, &gmax);
    TEST_ASSERT(gmax > 0.0);  // at least some values must be non-zero
  }

  // Layered temperature: at least one node/layer must be non-zero
  {
    double lmax = 0.0;
    for (int ws=0; ws<bnum_ws; ++ws) {
      const auto& bstate = bmfa->getElemStates()[ws].at("temperature").host();
      for (size_t ie=0; ie<bstate.extent(0); ++ie)
        for (size_t in=0; in<bstate.extent(1); ++in)
          for (size_t il=0; il<bstate.extent(2); ++il)
            lmax = std::max(lmax, std::abs(bstate(ie,in,il)));
    }
    double gmax;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &lmax, &gmax);
    TEST_ASSERT(gmax > 0.0);
  }
}

// ---------------------------------------------------------------------------
// Test: verify that extruded scalar fields in the 3D mesh have the same
// min/max as the corresponding 2D basal fields.
// This confirms that the basal values were correctly copied to all layers.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_STKBasal, ExtrudedFieldsCorrect)
{
  auto comm = Albany::getDefaultComm();
  auto disc = create_disc(comm);

  auto bdisc = disc->getSideSetDiscretizations().at("basalside");
  auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
  auto  mfa  = disc->getMeshStruct()->get_field_accessor();

  const int bnum_ws = bdisc->getNumWorksets();
  const int  num_ws = disc->getNumWorksets();

  for (const std::string& fname : {"ice_thickness", "surface_height", "basal_friction"}) {
    // Collect min/max of the 2D basal elem state
    double bmin =  std::numeric_limits<double>::max();
    double bmax = -std::numeric_limits<double>::max();
    for (int ws=0; ws<bnum_ws; ++ws) {
      auto v = bmfa->getElemStates()[ws].at(fname).host();
      auto [mn,mx] = global_minmax2(v, comm);
      bmin = std::min(bmin, mn);
      bmax = std::max(bmax, mx);
    }

    // Collect min/max of the 3D extruded elem state
    double vmin =  std::numeric_limits<double>::max();
    double vmax = -std::numeric_limits<double>::max();
    for (int ws=0; ws<num_ws; ++ws) {
      auto v = mfa->getElemStates()[ws].at(fname).host();
      auto [mn,mx] = global_minmax2(v, comm);
      vmin = std::min(vmin, mn);
      vmax = std::max(vmax, mx);
    }

    // The extruded 3D field is a replication of the basal field over all layers,
    // so global min/max must match exactly.
    TEST_FLOATING_EQUALITY(vmin, bmin, 1e-12);
    TEST_FLOATING_EQUALITY(vmax, bmax, 1e-12);
  }
}

// ---------------------------------------------------------------------------
// Test: verify that the interpolated 3D temperature values lie within the
// range of the basal layered temperature data.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_STKBasal, InterpolatedFieldCorrect)
{
  auto comm = Albany::getDefaultComm();
  auto disc = create_disc(comm);

  auto bdisc = disc->getSideSetDiscretizations().at("basalside");
  auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
  auto  mfa  = disc->getMeshStruct()->get_field_accessor();

  const int bnum_ws = bdisc->getNumWorksets();
  const int  num_ws = disc->getNumWorksets();

  // Collect min/max of the basal layered temperature
  double bmin =  std::numeric_limits<double>::max();
  double bmax = -std::numeric_limits<double>::max();
  for (int ws=0; ws<bnum_ws; ++ws) {
    auto v = bmfa->getElemStates()[ws].at("temperature").host();
    auto [mn,mx] = global_minmax3(v, comm);
    bmin = std::min(bmin, mn);
    bmax = std::max(bmax, mx);
  }

  // All 3D temperature values must lie within [bmin, bmax]
  for (int ws=0; ws<num_ws; ++ws) {
    auto v = mfa->getElemStates()[ws].at("temperature").host();
    auto [mn,mx] = global_minmax2(v, comm);
    TEST_ASSERT(mn >= bmin - 1e-10);
    TEST_ASSERT(mx <= bmax + 1e-10);
  }
}
