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
#include <Albany_Interface.hpp>
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
%include "Albany_Interface.hpp"
