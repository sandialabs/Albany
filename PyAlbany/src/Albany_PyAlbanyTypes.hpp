//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PYALBANYTYPES_H
#define ALBANY_PYALBANYTYPES_H

#include "TpetraCore_config.h"
#include "Tpetra_Map.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_MultiVector.hpp"

#if defined(HAVE_TPETRA_INST_INT_LONG_LONG)
#define PYALBANY_DOES_NOT_USE_DEEP_COPY
#endif

namespace PyAlbany
{
    /*
        PyTrilinos, which is used to simplify the Python/Albany interface, does not support the user to specify
        values of the template arguments.
        It is assumed that the global ordinal type is long long whereas Albany might use other global ordinal types.
        The following defined types are used to communicate information between Python and Albany.

        The local data of the vectors are deeply copied from those types to the ones supported by Albany if they
        are not consistent.
    */
    typedef Tpetra::Map<int, long long, Tpetra::Details::DefaultTypes::node_type> PyTrilinosMap;
    typedef Tpetra::Vector<double, int, long long, Tpetra::Details::DefaultTypes::node_type> PyTrilinosVector;
    typedef Tpetra::MultiVector<double, int, long long, Tpetra::Details::DefaultTypes::node_type> PyTrilinosMultiVector;
    typedef Tpetra::Export<int, long long, Tpetra::Details::DefaultTypes::node_type> PyTrilinosExport;
    typedef Tpetra::Import<int, long long, Tpetra::Details::DefaultTypes::node_type> PyTrilinosImport;
} // namespace PyAlbany

#endif // ALBANY_PYALBANYTYPES_H
