//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"

#include "Albany_config.h"
#ifdef ALBANY_OMEGAH
#include "Albany_CommUtils.hpp"
#include "Albany_Omegah.hpp"
#endif

int main( int argc, char* argv[] )
{
  // Note that the dtor for GlobalMPISession will call
  // Kokkos::finalize() but does NOT call Kokkos::initialize() for
  // any node type!
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize(argc,argv);
#ifdef ALBANY_OMEGAH
  Albany::init_omegah_lib (argc, argv, Albany::getDefaultComm());
#endif
  {
    Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
    out.setOutputToRootOnly(0);
    PHX::exec_space exec_space; 
    exec_space.print_configuration(out);
  }
  Teuchos::UnitTestRepository::setGloballyReduceTestResult(true);
  auto success = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

#ifdef ALBANY_OMEGAH
  Albany::finalize_omegah_lib ();
#endif

  return success;
}
