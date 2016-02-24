//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_DEFAULTREDUCEDBASISFACTORY_HPP
#define MOR_DEFAULTREDUCEDBASISFACTORY_HPP

#include "MOR_ReducedBasisFactory.hpp"

#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"

namespace MOR {

Teuchos::RCP<ReducedBasisFactory> defaultReducedBasisFactoryNew(const Epetra_Map &basisMap);

} // end namespace MOR

#endif /* MOR_DEFAULTREDUCEDBASISFACTORY_HPP */
