//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_UnitTestSession.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_CommUtils.hpp"
#include "DummyMesh.hpp"
#include "Albany_DiscretizationFactory.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <random>

TEUCHOS_UNIT_TEST (ExtrudedMesh, DiscFactoryCreateMesh)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto top_level_params = Teuchos::rcp(new Teuchos::ParameterList(""));
  auto disc_params = Teuchos::sublist(top_level_params,"Discretization");
  disc_params->set<std::string>("Method", "Extruded");
  disc_params->set<int>("NumLayers", 5);
  disc_params->set<int>("Number Of Time Derivatives", 0);
  disc_params->set<bool>("Columnwise Ordering", true);
  disc_params->sublist("Side Set Discretizations").set("Side Sets",Teuchos::Array<std::string>{"basalside"});
  auto& basal_params = disc_params->sublist("Side Set Discretizations").sublist("basalside");
  basal_params.set<std::string>("Method", "STK2D");
  basal_params.set<int>("1D Elements", 1);
  basal_params.set<int>("2D Elements", 1);
  basal_params.set<std::string>("Cell Topology", "Triangle");

  DiscretizationFactory factory(top_level_params,comm,false);

  auto mesh = factory.createMeshStruct(disc_params,comm,3);
  (void) mesh;
}

TEUCHOS_UNIT_TEST(ExtrudedMesh, Exceptions)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto& ts = UnitTestSession::instance();

  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine(ts.rng_seed);
  ipdf nlay_pdf (1,5);
  ipdf ne_x_pdf (1,5);

  // Create dummy basal mesh
  GO ne_x = ne_x_pdf(engine)*comm->getSize();
  auto numLayers = nlay_pdf(engine);
  Teuchos::broadcast(*comm,0,1,&ne_x);
  Teuchos::broadcast(*comm,0,1,&numLayers);

  auto mesh_2d = Teuchos::rcp(new DummyMesh(ne_x,comm));

  for (auto ordering : {LayeredMeshOrdering::COLUMN,LayeredMeshOrdering::LAYER}) {

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

    // Bad basal mesh dimension
    params->set<int>("NumLayers",1);
    auto bad_basal_mesh = Teuchos::rcp(new DummyMesh(ne_x,1,comm));
    TEST_THROW (Teuchos::rcp(new ExtrudedMesh(bad_basal_mesh,params,comm)),
                std::logic_error);
  }
}

TEUCHOS_UNIT_TEST(ExtrudedMesh, Counters)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto& ts = UnitTestSession::instance();

  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine(ts.rng_seed);
  ipdf nlay_pdf (1,5);
  ipdf ne_x_pdf (2*comm->getSize(),10*comm->getSize());

  // Create dummy basal mesh
  GO ne_x = ne_x_pdf(engine);
  auto numLayers = nlay_pdf(engine);
  auto mesh_2d = Teuchos::rcp(new DummyMesh(ne_x,comm));

  for (auto ordering : {LayeredMeshOrdering::COLUMN,LayeredMeshOrdering::LAYER}) {
    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set<int>("NumLayers",numLayers);
    params->set<int>("Workset Size",1000);
    params->set("Columnwise Ordering", ordering==LayeredMeshOrdering::COLUMN);

    auto mesh_3d = Teuchos::rcp(new ExtrudedMesh(mesh_2d,params,comm));
    mesh_3d->setFieldData(comm,Teuchos::null,{});
    mesh_3d->setBulkData(comm);

    TEST_EQUALITY (mesh_3d->get_num_local_elements(),mesh_2d->get_num_local_elements()*numLayers);
    TEST_EQUALITY (mesh_3d->get_num_local_nodes(),mesh_2d->get_num_local_nodes()*(numLayers+1));
    TEST_EQUALITY (mesh_3d->get_max_elem_gid(),(mesh_2d->get_max_elem_gid()+1)*numLayers);
    TEST_EQUALITY (mesh_3d->get_max_node_gid(),(mesh_2d->get_max_node_gid()+1)*(numLayers+1));
  }
}

TEUCHOS_UNIT_TEST(ExtrudedMesh, MeshParts)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  auto& ts = UnitTestSession::instance();

  using ipdf = std::uniform_int_distribution<int>;
  std::mt19937_64 engine(ts.rng_seed);
  ipdf nlay_pdf (1,5);
  ipdf ne_x_pdf (2*comm->getSize(),10*comm->getSize());

  // Create dummy basal mesh
  GO ne_x = ne_x_pdf(engine);
  auto numLayers = nlay_pdf(engine);
  Teuchos::broadcast(*comm,0,1,&ne_x);
  Teuchos::broadcast(*comm,0,1,&numLayers);

  auto mesh_2d = Teuchos::rcp(new DummyMesh(ne_x,comm));
  auto nsNames2d = mesh_2d->meshSpecs()[0]->nsNames;
  auto ssNames2d = mesh_2d->meshSpecs()[0]->ssNames;

  auto contains = [](const auto container, const auto val) {
    return std::find(container.begin(),container.end(),val)!=container.end();
  };
  for (auto ordering : {LayeredMeshOrdering::COLUMN,LayeredMeshOrdering::LAYER}) {
    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set<int>("NumLayers",numLayers);
    params->set<int>("Workset Size",1000);
    params->set("Columnwise Ordering", ordering==LayeredMeshOrdering::COLUMN);

    auto mesh_3d = Teuchos::rcp(new ExtrudedMesh(mesh_2d,params,comm));
    mesh_3d->setFieldData(comm,Teuchos::null,{});
    mesh_3d->setBulkData(comm);

    auto nsNames3d = mesh_3d->meshSpecs()[0]->nsNames;
    auto ssNames3d = mesh_3d->meshSpecs()[0]->ssNames;

    for (const auto& nsn : nsNames2d) {
      TEST_ASSERT (contains(nsNames3d,"basal_" + nsn));
      TEST_ASSERT (contains(nsNames3d,"extruded_" + nsn));
    }
    for (const auto& ssn : ssNames2d) {
      TEST_ASSERT (contains(ssNames3d,"extruded_" + ssn));
    }
    TEST_ASSERT (contains(nsNames3d,"bottom"));
    TEST_ASSERT (contains(nsNames3d,"top"));
    TEST_ASSERT (contains(nsNames3d,"lateral"));

    TEST_ASSERT (contains(ssNames3d,"basalside"));
    TEST_ASSERT (contains(ssNames3d,"upperside"));
    TEST_ASSERT (contains(ssNames3d,"lateralside"));
  }
}
