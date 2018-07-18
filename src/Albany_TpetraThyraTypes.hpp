//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_TPETRA_THYRA_TYPES_HPP
#define ALBANY_TPETRA_THYRA_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

// Get the converters
#include "Thyra_TpetraThyraWrappers.hpp"

typedef Thyra::TpetraOperatorVectorExtraction<ST, Tpetra_LO, Tpetra_GO, KokkosNode>   ConverterT;
typedef Thyra::TpetraMultiVector<ST,Tpetra_LO,Tpetra_GO,KokkosNode>                   Thyra_TpetraMultiVector;
typedef Thyra::TpetraVector<ST,Tpetra_LO,Tpetra_GO,KokkosNode>                        Thyra_TpetraVector;

#endif // ALBANY_TPETRA_THYRA_TYPES_HPP
