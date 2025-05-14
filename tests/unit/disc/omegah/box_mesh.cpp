//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

Teuchos::RCP<Albany::OmegahGenericMesh>
omegahGenericMesh(const Teuchos::RCP<Teuchos::ParameterList>& p) {
  return Teuchos::rcp(new Albany::OmegahGenericMesh(p));
}

TEUCHOS_UNIT_TEST(OmegahBoxMesh, 2D)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  Teuchos::Array<int> nelems(2);
  nelems[0] = 4;
  nelems[1] = 3;

  auto params = Teuchos::rcp(new Teuchos::ParameterList(""));
  params->set<std::string>("Mesh Creation Method","Box2D");

  // ---------------------------------- //
  //      Test exceptions handling      //
  // ---------------------------------- //

  // Missing required parameter
  TEST_THROW (omegahGenericMesh(params),std::logic_error);

  // Wrong dimensions
  params->set<Teuchos::Array<int>>("Number of Elements",Teuchos::Array<int>(3,1));
  TEST_THROW (omegahGenericMesh(params),std::logic_error);
  params->set<Teuchos::Array<int>>("Number of Elements",nelems);
  params->set<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(1,1));
  TEST_THROW (omegahGenericMesh(params),std::logic_error);

  // ---------------------------------- //
  //        Test normal oprations       //
  // ---------------------------------- //

  params->remove("Box Scaling");
  auto mesh = omegahGenericMesh(params);

  auto coords = mesh->coords_host();
  auto omegah_mesh = mesh->getOmegahMesh();
  auto owned = Omega_h::HostRead(omegah_mesh->owned(0));
  int my_count = 0;
  int count;
  for (int i=0; i<owned.size(); ++i) {
    if (owned[i]) { ++my_count; }
  }
  Teuchos::reduceAll<int, int>(*comm, Teuchos::REDUCE_SUM, 1, &my_count, &count);

  TEST_EQUALITY_CONST(count,(nelems[0]+1)*(nelems[1]+1));
}

TEUCHOS_UNIT_TEST(OmegahBoxMesh, 1D)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  Teuchos::Array<int> nelems(1);
  nelems[0] = 4;

  auto params = Teuchos::rcp(new Teuchos::ParameterList(""));
  params->set<std::string>("Mesh Creation Method","Box1D");

  // ---------------------------------- //
  //      Test exceptions handling      //
  // ---------------------------------- //

  // Missing required parameter
  TEST_THROW (omegahGenericMesh(params),std::logic_error);

  // Wrong dimensions
  params->set<Teuchos::Array<int>>("Number of Elements",Teuchos::Array<int>(2,1));
  TEST_THROW (omegahGenericMesh(params),std::logic_error);
  params->set<Teuchos::Array<int>>("Number of Elements",nelems);
  params->set<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(3,1));
  TEST_THROW (omegahGenericMesh(params),std::logic_error);

  // ---------------------------------- //
  //        Test normal oprations       //
  // ---------------------------------- //

  params->remove("Box Scaling");
  auto mesh = omegahGenericMesh(params);

  auto coords = mesh->coords_host();
  auto omegah_mesh = mesh->getOmegahMesh();
  auto owned = Omega_h::HostRead(omegah_mesh->owned(0));
  int my_count = 0;
  int count;
  for (int i=0; i<owned.size(); ++i) {
    if (owned[i]) { ++my_count; }
  }
  Teuchos::reduceAll<int, int>(*comm, Teuchos::REDUCE_SUM, 1, &my_count, &count);

  TEST_EQUALITY_CONST(count,(nelems[0]+1));
}
