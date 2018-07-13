//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DATA_TYPES_HPP
#define ALBANY_DATA_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

// Get scalar and ordinal types
#include "Albany_ScalarOrdinalTypes.hpp"

// Get all Tpetra types
#include "Albany_TpetraTypes.hpp"

// Get all Sacado types (and helpers)
#include "Albany_SacadoTypes.hpp"

// Get all Thyra types
#include "Albany_ThyraTypes.hpp"

// Get the Thyra-Tpetra converter
#include "Albany_TpetraThyraTypes.hpp"

// Get all comm types
#include "Albany_CommTypes.hpp"

// Code macros to support deprecated warnings
#ifdef ALBANY_ENABLE_DEPRECATED
#  if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define ALBANY_DEPRECATED  __attribute__((__deprecated__))
#  else
#    define ALBANY_DEPRECATED
#  endif
#endif

#endif // ALBANY_DATA_TYPES_HPP
