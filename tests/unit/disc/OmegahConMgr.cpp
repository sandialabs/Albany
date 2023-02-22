//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"
#include "OmegahConnManager.hpp"
#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

// I find assert-style checks more intuitive in unit tests.
// So throw if the input condition is false:
#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto comm = Albany::getDefaultComm();

  auto lib = Omega_h::Library();
  auto commPtr = Omega_h::Comm(lib, comm);
  auto mesh = Omega_h::build_box(commPtr, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 2, false);

  auto conn_mgr = Teuchos::rcp(new OmegahConnManager(mesh)); //FIXME - this will fail

  // Silence compiler warnings due to unused stuff from Teuchos testing framework.
  (void) out;
  (void) success;
}
