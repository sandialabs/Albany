//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_SolverFactory.hpp"
#include "Albany_DakotaT.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

// Global variable that denotes this is the Tpetra executable
bool TpetraBuild = false;

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Kokkos::initialize(argc, argv);
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  try {
    Teuchos::RCP<Teuchos::Time> totalTime =
      Teuchos::TimeMonitor::getNewTimer("AlbanyDakotaT: ***Total Time***");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer

    status += Albany_DakotaT(argc, argv);

    // Regression comparisons fopr Dakota runs only valid on Proc 0.
    if (mpiSession.getRank() > 0) status=0;
    else *out << "\nNumber of Failed Comparisons: " << status << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(std::cout, false, true, false);
  Kokkos::finalize_all();
  return status;
}
