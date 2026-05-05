//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests for the Extruded discretization using an Omega_h basal mesh.
// These tests use the GIS (Greenland Ice Sheet) populated .osh mesh
// (gis_unstruct_basal_populated.osh) as the basal mesh.  Because Exodus and
// Omega_h layered-field formats differ, temperature is left out of this test;
// scalar fields (ice_thickness, surface_height, basal_friction) are loaded
// from constant values so that the extrusion logic is exercised reliably.
//
// What is tested:
//  - the extruded disc can be created with an Omega_h basal mesh
//  - scalar fields are correctly extruded to the 3D mesh
//  - expected mesh parts (node sets and side sets) are present

#include "Albany_UnitTestSession.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "../ExtrudedDiscTestUtils.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <algorithm>
#include <limits>

// The GIS basal mesh has 2D triangular elements; the extruded 3D mesh has wedge elements.
static constexpr int num_layers = 5;  // number of vertical layers in extruded mesh
static constexpr int neq        = 2;  // number of equations (arbitrary, just to build disc)
static constexpr int nBNodes    = 3;  // nodes per basal triangle element
static constexpr int nVNodes    = 6;  // nodes per extruded wedge element

static const std::string osh_file = "./gis_unstruct_basal_populated.osh";

// Known constant values used to fill the basal fields
static constexpr double H_value    = 1000.0;  // ice thickness (m)
static constexpr double zs_value   = 2000.0;  // surface height (m)
static constexpr double beta_value =    1.0;  // basal friction

// Name of the lateral side set in the .osh file (used to mark the part in Omega_h)
static const std::string lateral_part = "lateralside";

namespace {

// Create an extruded discretization with an Omega_h basal mesh loaded from the GIS .osh file.
// Scalar fields are filled with fixed constants ("Field Origin: File", "Field Value").
// The extruded disc extrudes all three scalar fields to the 3D mesh.
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

  // Side-set discretization: only basalside
  disc_params->sublist("Side Set Discretizations")
             .set("Side Sets", Teuchos::Array<std::string>{"basalside"});

  // Basal mesh: Omega_h .osh file
  auto& basal_params = disc_params->sublist("Side Set Discretizations").sublist("basalside");
  basal_params.set<std::string>("Method", "Omegah");
  basal_params.set<int>("Number Of Time Derivatives", 0);
  basal_params.set<std::string>("Mesh Creation Method", "OshFile");
  basal_params.set<std::string>("Input Filename", osh_file);
  // Mark the lateral boundary so it appears in the basal mesh side sets
  basal_params.set("Mark Parts", Teuchos::Array<std::string>{lateral_part});

  // Scalar fields filled with known constants
  auto& basal_req = basal_params.sublist("Required Fields Info");
  basal_req.set<int>("Number Of Fields", 3);

  auto& H_req   = basal_req.sublist("Field 0");
  H_req.set<std::string>("Field Name",   "ice_thickness");
  H_req.set<std::string>("Field Type",   "Node Scalar");
  H_req.set<std::string>("Field Origin", "File");
  H_req.set<double>("Field Value", H_value);

  auto& zs_req  = basal_req.sublist("Field 1");
  zs_req.set<std::string>("Field Name",   "surface_height");
  zs_req.set<std::string>("Field Type",   "Node Scalar");
  zs_req.set<std::string>("Field Origin", "File");
  zs_req.set<double>("Field Value", zs_value);

  auto& beta_req = basal_req.sublist("Field 2");
  beta_req.set<std::string>("Field Name",   "basal_friction");
  beta_req.set<std::string>("Field Type",   "Node Scalar");
  beta_req.set<std::string>("Field Origin", "File");
  beta_req.set<double>("Field Value", beta_value);

  // ---------------------------------------------------------------------------
  // Basal StateInfoStruct: describes fields stored on the 2D basal mesh.
  // dim[0] is a placeholder (replaced by workset size); dim[1] = nodes per
  // basal element (= 3 for triangles).
  // ---------------------------------------------------------------------------
  std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct>> ss_sis;
  auto& bsis = ss_sis["basalside"] = Teuchos::rcp(new Albany::StateInfoStruct());

  auto bst_H    = Teuchos::rcp(new Albany::StateStruct(
      "ice_thickness",  Albany::StateStruct::NodalDataToElemNode, {1, nBNodes}));
  auto bst_zs   = Teuchos::rcp(new Albany::StateStruct(
      "surface_height", Albany::StateStruct::NodalDataToElemNode, {1, nBNodes}));
  auto bst_beta = Teuchos::rcp(new Albany::StateStruct(
      "basal_friction", Albany::StateStruct::NodalDataToElemNode, {1, nBNodes}));

  bsis->push_back(bst_H);
  bsis->push_back(bst_zs);
  bsis->push_back(bst_beta);

  // ---------------------------------------------------------------------------
  // Volume StateInfoStruct: describes fields on the 3D extruded mesh (wedge6).
  // dim[1] = nodes per wedge element (= 6).
  // ---------------------------------------------------------------------------
  auto sis = Teuchos::rcp(new Albany::StateInfoStruct());

  auto st_H    = Teuchos::rcp(new Albany::StateStruct(
      "ice_thickness",  Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));
  auto st_zs   = Teuchos::rcp(new Albany::StateStruct(
      "surface_height", Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));
  auto st_beta = Teuchos::rcp(new Albany::StateStruct(
      "basal_friction", Albany::StateStruct::NodalDataToElemNode, {1, nVNodes}));

  sis->push_back(st_H);
  sis->push_back(st_zs);
  sis->push_back(st_beta);

  Albany::DiscretizationFactory factory(top_level_params, comm, false);
  factory.createMeshSpecs();
  return factory.createDiscretization(neq, {}, sis, ss_sis);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test: verify that the expected mesh parts are created in the extruded disc.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_OmegahBasal, MeshPartsCreated)
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
// Test: verify that the basal fields are loaded with the expected constant
// values.  Since we used "Field Value", every node must equal the constant.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_OmegahBasal, FieldsLoaded)
{
  auto comm = Albany::getDefaultComm();
  auto disc = create_disc(comm);

  auto bdisc = disc->getSideSetDiscretizations().at("basalside");
  auto bmfa  = bdisc->getMeshStruct()->get_field_accessor();
  const int bnum_ws = bdisc->getNumWorksets();

  const std::pair<const std::string, double> checks[] = {
    {"ice_thickness",  H_value},
    {"surface_height", zs_value},
    {"basal_friction", beta_value},
  };

  for (const auto& [fname, expected] : checks) {
    for (int ws=0; ws<bnum_ws; ++ws) {
      const auto& bstate = bmfa->getElemStates()[ws].at(fname).host();
      for (size_t ie=0; ie<bstate.extent(0); ++ie)
        for (size_t in=0; in<bstate.extent(1); ++in)
          TEST_FLOATING_EQUALITY(bstate(ie,in), expected, 1e-12);
    }
  }
}

// ---------------------------------------------------------------------------
// Test: verify that extruded scalar fields in the 3D mesh match the
// constant basal values at every node of every layer.
// ---------------------------------------------------------------------------
TEUCHOS_UNIT_TEST(ExtrudedDisc_OmegahBasal, ExtrudedFieldsCorrect)
{
  auto comm = Albany::getDefaultComm();
  auto disc = create_disc(comm);

  auto mfa      = disc->getMeshStruct()->get_field_accessor();
  auto bdisc    = disc->getSideSetDiscretizations().at("basalside");
  auto bmfa     = bdisc->getMeshStruct()->get_field_accessor();

  const int bnum_ws = bdisc->getNumWorksets();
  const int  num_ws = disc->getNumWorksets();

  const std::pair<const std::string, double> checks[] = {
    {"ice_thickness",  H_value},
    {"surface_height", zs_value},
    {"basal_friction", beta_value},
  };

  for (const auto& [fname, expected] : checks) {
    // Basal: accumulate local min/max across all worksets, then do a single global reduction.
    double lbmin =  std::numeric_limits<double>::max();
    double lbmax = -std::numeric_limits<double>::max();
    for (int ws=0; ws<bnum_ws; ++ws) {
      auto v = bmfa->getElemStates()[ws].at(fname).host();
      auto [mn, mx] = ExtrudedDiscTestUtils::local_minmax2(v);
      lbmin = std::min(lbmin, mn);
      lbmax = std::max(lbmax, mx);
    }
    double bmin, bmax;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &lbmin, &bmin);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &lbmax, &bmax);
    TEST_FLOATING_EQUALITY(bmin, expected, 1e-12);
    TEST_FLOATING_EQUALITY(bmax, expected, 1e-12);

    // 3D: accumulate local min/max across all worksets, then do a single global reduction.
    double lvmin =  std::numeric_limits<double>::max();
    double lvmax = -std::numeric_limits<double>::max();
    for (int ws=0; ws<num_ws; ++ws) {
      auto v = mfa->getElemStates()[ws].at(fname).host();
      auto [mn, mx] = ExtrudedDiscTestUtils::local_minmax2(v);
      lvmin = std::min(lvmin, mn);
      lvmax = std::max(lvmax, mx);
    }
    double vmin, vmax;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &lvmin, &vmin);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &lvmax, &vmax);
    TEST_FLOATING_EQUALITY(vmin, expected, 1e-12);
    TEST_FLOATING_EQUALITY(vmax, expected, 1e-12);
  }
}
