//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"

// This is the same as the Trilinos/packages/phalanx/test/Utilities/Phalanx_UnitTestMain.cpp
// and should be kept "in sync" with that as things evolve

int main( int argc, char* argv[] )
{
  // Note that the dtor for GlobalMPISession will call
  // Kokkos::finalize_all() but does NOT call Kokkos::initialize() for
  // any node type!
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize(argc,argv);
  {
    Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
    out.setOutputToRootOnly(0);
    PHX::exec_space::print_configuration(out);
  }
  Teuchos::UnitTestRepository::setGloballyReduceTestResult(true);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}
