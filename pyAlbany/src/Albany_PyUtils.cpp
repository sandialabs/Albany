//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PyUtils.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GitVersion.h"

#include <mpi.h>
#include <Teuchos_DefaultMpiComm.hpp>

#include <cstdlib>
#include <stdexcept>
#include <time.h>

#include "MatrixMarket_Tpetra.hpp"
#include "Teuchos_TestForException.hpp"
#include "Kokkos_Macros.hpp"

// For stack trace
#include <execinfo.h>
#include <cstdarg>
#include <cstdio>

#include "Teuchos_DefaultSerialComm.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Details_MpiTypeTraits.hpp"

namespace PyAlbany
{
  void
  PrintPyHeader(std::ostream &os)
  {
    os << R"(*********************************************************************************)" << std::endl;
    os << R"(**  ______   __  __   ______   __       ______   ______   __   __   __  __     **)" << std::endl;
    os << R"(** /\  == \ /\ \_\ \ /\  __ \ /\ \     /\  == \ /\  __ \ /\ "-.\ \ /\ \_\ \    **)" << std::endl;
    os << R"(** \ \  __/ \ \____ \\ \  __ \\ \ \____\ \  __< \ \  __ \\ \ \-.  \\ \____ \   **)" << std::endl;
    os << R"(**  \ \_\    \/\_____\\ \_\ \_\\ \_____\\ \_____\\ \_\ \_\\ \_\\"\_\\/\_____\  **)" << std::endl;
    os << R"(**   \/_/     \/_____/ \/_/\/_/ \/_____/ \/_____/ \/_/\/_/ \/_/ \/_/ \/_____/  **)" << std::endl;
    os << R"(**                                                                             **)" << std::endl;
    os << R"(*********************************************************************************)" << std::endl;
    os << R"(** Trilinos git commit id - )" << ALBANY_TRILINOS_GIT_COMMIT_ID << std::endl;
    os << R"(** Albany git branch ------ )" << ALBANY_GIT_BRANCH << std::endl;
    os << R"(** Albany git commit id --- )" << ALBANY_GIT_COMMIT_ID << std::endl;
    os << R"(** Albany cxx compiler ---- )" << CMAKE_CXX_COMPILER_ID << " " << CMAKE_CXX_COMPILER_VERSION << std::endl;

#ifdef KOKKOS_COMPILER_CUDA_VERSION
    os << R"(** Albany cuda compiler --- Cuda )" << KOKKOS_COMPILER_CUDA_VERSION << std::endl;
#endif

    // Print start time
    time_t rawtime;
    time(&rawtime);
    struct tm *timeinfo = localtime(&rawtime);
    char buffer[80];
    strftime(buffer, 80, "%F at %T", timeinfo);
    os << R"(** Simulation start time -- )" << buffer << std::endl;
    os << R"(***************************************************************)" << std::endl;
  }
} // namespace PyAlbany
