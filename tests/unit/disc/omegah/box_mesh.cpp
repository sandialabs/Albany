//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_OmegahBoxMesh.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

TEUCHOS_UNIT_TEST(OmegahBoxMesh, 2D)
{
  using namespace Albany;

  auto comm = getDefaultComm();

  Teuchos::Array<int> nelems(2);
  nelems[0] = 4;
  nelems[1] = 3;
  const int num_params = 2;

  auto params = Teuchos::rcp(new Teuchos::ParameterList(""));

  // ---------------------------------- //
  //      Test exceptions handling      //
  // ---------------------------------- //

  // Missing required parameter
  TEST_THROW (OmegahBoxMesh<2>(params,comm,num_params),std::logic_error);

  // Wrong dimensions
  params->set<Teuchos::Array<int>>("Number of Elements",Teuchos::Array<int>(3,1));
  TEST_THROW (OmegahBoxMesh<2>(params,comm,num_params),std::logic_error);
  params->set<Teuchos::Array<int>>("Number of Elements",nelems);
  params->set<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(1,1));
  TEST_THROW (OmegahBoxMesh<2>(params,comm,num_params),std::logic_error);

  // ---------------------------------- //
  //        Test normal oprations       //
  // ---------------------------------- //

  params->remove("Box Scaling");
  auto mesh = Teuchos::rcp(new OmegahBoxMesh<2>(params,comm,num_params));

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
  const int num_params = 1;

  auto params = Teuchos::rcp(new Teuchos::ParameterList(""));

  // ---------------------------------- //
  //      Test exceptions handling      //
  // ---------------------------------- //

  // Missing required parameter
  TEST_THROW (OmegahBoxMesh<1>(params,comm,num_params),std::logic_error);

  // Wrong dimensions
  params->set<Teuchos::Array<int>>("Number of Elements",Teuchos::Array<int>(2,1));
  TEST_THROW (OmegahBoxMesh<1>(params,comm,num_params),std::logic_error);
  params->set<Teuchos::Array<int>>("Number of Elements",nelems);
  params->set<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(3,1));
  TEST_THROW (OmegahBoxMesh<1>(params,comm,num_params),std::logic_error);

  // ---------------------------------- //
  //        Test normal oprations       //
  // ---------------------------------- //

  params->remove("Box Scaling");
  auto mesh = Teuchos::rcp(new OmegahBoxMesh<1>(params,comm,num_params));

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
