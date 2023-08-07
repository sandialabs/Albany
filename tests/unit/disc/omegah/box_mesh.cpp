//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

TEUCHOS_UNIT_TEST(OmegahBoxMesh, 2D)
{
  using namespace Albany;

  build_type (BuildType::Tpetra);
  auto comm = getDefaultComm();

  const int nelemx = 4;
  const int nelemy = 3;
  const int nelemz = 2;
  const int num_params = 2;

  auto params = Teuchos::rcp(new Teuchos::ParameterList(""));
  params.set<int>("1D Elements",nelemx);
  params.set<int>("2D Elements",nelemy);
  params.set<int>("3D Elements",nelemz);

  auto mesh = Teuchos::rcp(new OmegahBoxMesh<2>(params,comm,num_params));

  auto coords = mesh->coords_host();
  TEST_EQUALITY_CONST(coords.size(),(nelemx+1)*(nelemy+1));
}
