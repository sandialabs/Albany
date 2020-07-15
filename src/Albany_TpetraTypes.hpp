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

typedef int Tpetra_LO;
#if defined( HAVE_TPETRA_INST_INT_LONG_LONG )
typedef long long Tpetra_GO;
#elif defined( HAVE_TPETRA_INST_INT_LONG )
static_assert(sizeof(long) == sizeof(GO),
    "Tpetra's biggest enabled GlobalOrdinal is long but thats not 64 bit");
typedef long Tpetra_GO;
#elif defined( HAVE_TPETRA_INST_INT_UNSIGNED_LONG )
static_assert(sizeof(unsigned long) == sizeof(GO),
    "Tpetra's biggest enabled GlobalOrdinal is unsigned long but thats not 64 bit");
typedef unsigned long Tpetra_GO;
#else
#error "Albany needs a 64-bit GlobalOrdinal enabled in Tpetra"
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

#endif // ALBANY_TPETRA_TYPES_HPP
