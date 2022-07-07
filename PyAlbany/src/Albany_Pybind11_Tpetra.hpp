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


/**
 * \brief createRCPPyMapEmpty function
 * 
 * This function is used to create an RCP to an empty Tpetra_Map.
 */
RCP_PyMap createRCPPyMapEmpty();

/**
 * \brief createRCPPyMap function
 * 
 * This function is used to create an RCP to a Tpetra_Map
 * given a number of global elements numGlobalEl, a number
 * of local elements numMyEl, an index base indexBase,
 * and an RCP to a Teuchos::Comm<int>.
 */
RCP_PyMap createRCPPyMap(int numGlobalEl, int numMyEl, int indexBase, RCP_Teuchos_Comm_PyAlbany comm );

/**
 * \brief createRCPPyMapFromView function
 * 
 * This function is used to create an RCP to a Tpetra_Map
 * given a number of global elements numGlobalEl, a numpy
 * array with the index list indexList, an index base indexBase,
 * and an RCP to a Teuchos::Comm<int>.
 */
RCP_PyMap createRCPPyMapFromView(int numGlobalEl, pybind11::array_t<int> indexList, int indexBase, RCP_Teuchos_Comm_PyAlbany comm );

/**
 * \brief createRCPPyVectorEmpty function
 * 
 * This function is used to create an RCP to an empty Tpetra_Vector.
 */
RCP_PyVector createRCPPyVectorEmpty();

/**
 * \brief createRCPPyVector1 function
 * 
 * This function is used to create an RCP to a Tpetra_Vector
 * given an RCP to a Tpetra_Map and a boolean to zero out the entries.
 */
RCP_PyVector createRCPPyVector1(RCP_PyMap &map, const bool zeroOut);

/**
 * \brief createRCPPyVector2 function
 * 
 * This function is used to create an RCP to a Tpetra_Vector
 * given an RCP to a const Tpetra_Map and a boolean to zero out the entries.
 */
RCP_PyVector createRCPPyVector2(RCP_ConstPyMap &map, const bool zeroOut);

/**
 * \brief createRCPPyMultiVectorEmpty function
 * 
 * This function is used to create an RCP to an empty Tpetra_MultiVector.
 */
RCP_PyMultiVector createRCPPyMultiVectorEmpty();

/**
 * \brief createRCPPyMultiVector1 function
 * 
 * This function is used to create an RCP to a Tpetra_MultiVector
 * given an RCP to a Tpetra_Map, a number of columns, and a boolean to zero out the entries.
 */
RCP_PyMultiVector createRCPPyMultiVector1(RCP_PyMap &map, const int n_cols, const bool zeroOut);

/**
 * \brief createRCPPyMultcreateRCPPyMultiVector2iVector1 function
 * 
 * This function is used to create an RCP to a Tpetra_MultiVector
 * given an RCP to a const Tpetra_Map, a number of columns, and a boolean to zero out the entries.
 */
RCP_PyMultiVector createRCPPyMultiVector2(RCP_ConstPyMap &map, const int n_cols, const bool zeroOut);

/**
 * \brief getLocalViewHost function
 * 
 * This function returns a local view on host of a Tpetra_Vector.
 */
pybind11::array_t<ST> getLocalViewHost(RCP_PyVector &vector);

/**
 * \brief getLocalViewHost function
 * 
 * This function returns a local view on host of a Tpetra_MultiVector.
 */
pybind11::array_t<ST> getLocalViewHost(RCP_PyMultiVector &mvector);

/**
 * \brief setLocalViewHost function
 * 
 * This function sets the local view of a Tpetra_Vector.
 */
void setLocalViewHost(RCP_PyVector &vector, pybind11::array_t<double> input);

/**
 * \brief setLocalViewHost function
 * 
 * This function sets the local view of a Tpetra_MultiVector.
 */
void setLocalViewHost(RCP_PyMultiVector &mvector, pybind11::array_t<double> input);

/**
 * \brief getRemoteIndexList function
 * 
 * This function gets the remote index list of a Tpetra_Map.
 */
pybind11::tuple getRemoteIndexList(RCP_ConstPyMap map, pybind11::array_t<Tpetra_GO> globalIndexes);

void pyalbany_map(pybind11::module &m);
void pyalbany_vector(pybind11::module &m);
void pyalbany_mvector(pybind11::module &m);

#endif
