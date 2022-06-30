//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_TPETRA_TYPES_HPP
#define ALBANY_TPETRA_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

// Get the scalar and ordinal types
#include "Albany_ScalarOrdinalTypes.hpp"

// Get the kokkos types
#include "Albany_KokkosTypes.hpp"

// Tpetra includes
// Ignore annoying warnings that pollute compile log so extensively
#include "TpetraCore_config.h"
#include "Tpetra_Map.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_FECrsGraph.hpp"
#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_DistObject.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"

#ifndef HAVE_TPETRA_INST_DOUBLE
#error "Albany needs Tpetra to enable double as a Scalar type"
#endif

using Tpetra_LO = int;
// NOTE: this ifdef logic allows Tpetra_GO to be the *exact same type*
//       as panzer::GlobalOrdinal. In turn, panzer is keeping the first
//       branch of the if only for backward compatibility with some
//       deprecated code. As soon as that code in PanzerCore_config.hpp
//       is removed, we can keep the one using DefaulTypes.
#if defined(HAVE_TPETRA_INST_INT_LONG_LONG)
using Tpetra_GO = long long int;  // NOTE: long long is *guaranteed* to be >=64 bits
#else
using Tpetra_GO = Tpetra::Details::DefaultTypes::global_ordinal_type;
static_assert(sizeof(Tpetra_GO) == sizeof(GO),
    "Tpetra's default global ordinal is not 64 bit");
#endif

typedef Tpetra::Map<Tpetra_LO, Tpetra_GO, KokkosNode>                 Tpetra_Map;
typedef Tpetra::Export<Tpetra_LO, Tpetra_GO, KokkosNode>              Tpetra_Export;
typedef Tpetra::Import<Tpetra_LO, Tpetra_GO, KokkosNode>              Tpetra_Import;
typedef Tpetra::CrsGraph<Tpetra_LO, Tpetra_GO, KokkosNode>            Tpetra_CrsGraph;
typedef Tpetra::CrsMatrix<ST, Tpetra_LO, Tpetra_GO, KokkosNode>       Tpetra_CrsMatrix;
typedef Tpetra::FECrsGraph<Tpetra_LO, Tpetra_GO, KokkosNode>          Tpetra_FECrsGraph;
typedef Tpetra::FECrsMatrix<ST, Tpetra_LO, Tpetra_GO, KokkosNode>     Tpetra_FECrsMatrix;
typedef Tpetra::RowMatrix<ST, Tpetra_LO, Tpetra_GO, KokkosNode>       Tpetra_RowMatrix;
typedef Tpetra::Operator<ST, Tpetra_LO, Tpetra_GO, KokkosNode>        Tpetra_Operator;
typedef Tpetra::Vector<ST, Tpetra_LO, Tpetra_GO, KokkosNode>          Tpetra_Vector;
typedef Tpetra::MultiVector<ST, Tpetra_LO, Tpetra_GO, KokkosNode>     Tpetra_MultiVector;

// Make sure our Kokkos types are compatible with what tpetra uses
// This will make it easier to debug problems such as the one in issue #612 on GitHub
static_assert (std::is_same<typename Tpetra_CrsGraph::local_graph_device_type,Albany::DeviceLocalGraph>::value,
               "Error! Tpetra's local_graph_device_type does not match Albany's DeviceLocalGraph.\n"
               "       Most likely there was a change in Tpetra, and you need to update Albany_KokkosTypes.hpp.\n");
static_assert (std::is_same<typename Tpetra_CrsMatrix::local_matrix_device_type,Albany::DeviceLocalMatrix<ST>>::value,
               "Error! Tpetra's local_matrix_device_type does not match Albany's DeviceLocalMatrix.\n"
               "       Most likely there was a change in Tpetra, and you need to update Albany_KokkosTypes.hpp.\n");

#endif // ALBANY_TPETRA_TYPES_HPP
