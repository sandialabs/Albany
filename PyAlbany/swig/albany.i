//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// SWIG input file of PyAlbany

%module(docstring="'wpyalbany'") wpyalbany
%{

#include <string>
#include <sstream>
#include <typeinfo>

#include <Albany_PyAlbanyTypes.hpp>
#define SWIG_FILE_WITH_INIT
#include <Albany_Interface.hpp>
%}
// ----------- Numpy -----------
%include "numpy.i"

%init %{
    import_array();
%}

// ----------- PyTrilinos ------------
%include "Teuchos_RCP_typemaps.i"
%include "Teuchos.i"
%include "Tpetra.i"

// ----------- String ------------
%include "std_string.i"
using std::string;

///////////////////////////
// Teuchos::Time support //
///////////////////////////
%teuchos_rcp(Teuchos::StackedTimer)
%include "Teuchos_StackedTimer.hpp"

// ---------- Shared_ptr ----------
%teuchos_rcp(PyAlbany::PyParallelEnv)
%teuchos_rcp(PyAlbany::PyProblem)

%include "Albany_PyAlbanyTypes.hpp"
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* C, int n, int m)}
%include "Albany_Interface.hpp"
