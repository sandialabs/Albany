//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_PARALLELENV_H
#define PYALBANY_PARALLELENV_H

#include "Albany_Pybind11_Comm.hpp"
#include "Albany_Interface.hpp"

#include "Albany_Pybind11_Include.hpp"

using PyParallelEnv = PyAlbany::PyParallelEnv;

void pyalbany_parallelenv(pybind11::module &m);

#endif
