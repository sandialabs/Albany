//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_TPETRA_H
#define PYALBANY_TPETRA_H

#include "Albany_TpetraTypes.hpp"
#include "Albany_Pybind11_Comm.hpp"

#include "Albany_Pybind11_Include.hpp"

using PyMap = Tpetra_Map;
using RCP_PyMap = Teuchos::RCP<PyMap>;
using RCP_ConstPyMap = Teuchos::RCP<const Tpetra_Map>;
using RCP_PyVector = Teuchos::RCP<Tpetra_Vector>;
using RCP_PyMultiVector = Teuchos::RCP<Tpetra_MultiVector>;
using RCP_PyCrsMatrix = Teuchos::RCP<Tpetra_CrsMatrix>;

void pyalbany_map(pybind11::module &m);
void pyalbany_vector(pybind11::module &m);
void pyalbany_mvector(pybind11::module &m);
void pyalbany_crsmatrix(pybind11::module &m);

#endif
