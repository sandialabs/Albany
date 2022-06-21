//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_COMM_H
#define PYALBANY_COMM_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultComm.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>
#include <mpi4py/mpi4py.h>

using Teuchos_Comm_PyAlbany = Teuchos::Comm<int>;
using RCP_Teuchos_Comm_PyAlbany = Teuchos::RCP<const Teuchos_Comm_PyAlbany >;

void pyalbany_comm(pybind11::module &m);

#endif
