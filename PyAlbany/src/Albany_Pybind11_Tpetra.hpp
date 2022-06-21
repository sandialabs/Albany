//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_TPETRA_H
#define PYALBANY_TPETRA_H

#include "Albany_TpetraTypes.hpp"
#include "Albany_Pybind11_Comm.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using RCP_PyMap = Teuchos::RCP<Tpetra_Map>;
using RCP_ConstPyMap = Teuchos::RCP<const Tpetra_Map>;
using RCP_PyVector = Teuchos::RCP<Tpetra_Vector>;
using RCP_PyMultiVector = Teuchos::RCP<Tpetra_MultiVector>;


RCP_PyMap createRCPPyMapEmpty();

RCP_PyMap createRCPPyMap(int numGlobalEl, int numMyEl, int indexBase, RCP_Teuchos_Comm_PyAlbany comm );

RCP_PyMap createRCPPyMapFromView(int numGlobalEl, pybind11::array_t<int> indexList, int indexBase, RCP_Teuchos_Comm_PyAlbany comm );

RCP_PyVector createRCPPyVectorEmpty();

RCP_PyVector createRCPPyVector1(RCP_PyMap &map, const bool zeroOut);

RCP_PyVector createRCPPyVector2(RCP_ConstPyMap &map, const bool zeroOut);

RCP_PyMultiVector createRCPPyMultiVectorEmpty();

RCP_PyMultiVector createRCPPyMultiVector1(RCP_PyMap &map, const int n_cols, const bool zeroOut);

RCP_PyMultiVector createRCPPyMultiVector2(RCP_ConstPyMap &map, const int n_cols, const bool zeroOut);

pybind11::array_t<ST> getLocalViewHost(RCP_PyVector &vector);

pybind11::array_t<ST> getLocalViewHost(RCP_PyMultiVector &mvector);

void setLocalViewHost(RCP_PyVector &vector, pybind11::array_t<double> input);

void setLocalViewHost(RCP_PyMultiVector &mvector, pybind11::array_t<double> input);

pybind11::tuple getRemoteIndexList(RCP_ConstPyMap map, pybind11::array_t<Tpetra_GO> globalIndexes);

void pyalbany_map(pybind11::module &m);
void pyalbany_vector(pybind11::module &m);
void pyalbany_mvector(pybind11::module &m);

#endif
