//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Kokkos_Core.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_UnitTestRepository.hpp"

int
main(int argc, char* argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize();

  return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
  Kokkos::finalize();
}
