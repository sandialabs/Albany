//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Epetra_MpiComm.h>

#include "Albany_Layouts.hpp"

#include "peridigm/PeridigmManager.hpp"


namespace {

  TEUCHOS_UNIT_TEST( Peridigm, Instantiation )
  {
    // construct a PeridigmManager
    LCM::PeridigmManager peridigm_manager;

    TEST_ASSERT( true );
  }
} // namespace
