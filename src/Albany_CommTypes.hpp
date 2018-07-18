//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_COMM_TYPES_HPP
#define ALBANY_COMM_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

// Get Teuchos Comm type
#include "Teuchos_DefaultComm.hpp"

#ifdef ALBANY_MPI
  using Albany_MPI_Comm = MPI_Comm;

  #define Albany_MPI_COMM_WORLD MPI_COMM_WORLD
  #define Albany_MPI_COMM_NULL MPI_COMM_NULL
#else
  using Albany_MPI_Comm = int;

  #define Albany_MPI_COMM_WORLD 0  // This is compatible with Dakota
  #define Albany_MPI_COMM_NULL 99
#endif

// Teuchos comm typedef
typedef Teuchos::Comm<int>  Teuchos_Comm;

#endif // ALBANY_COMM_TYPES_HPP
