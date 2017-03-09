//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#if defined(ALBANY_CHECK_FPE)
#include <math.h>
#include <xmmintrin.h>
#endif

#if defined(ALBANY_FLUSH_DENORMALS)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"
#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "Kokkos_Core.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx.hpp"
#include "Piro_PerformSolve.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

// Global variable that denotes this is a Tpetra executable
bool TpetraBuild = false;

int main(int ac, char *av[]) {

  // 0 = pass, failures are incremented
  int
  status{0};

  bool
  success{true};

  return status;
}
