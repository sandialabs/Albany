//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_EPETRALOCALMAPMVMATRIXMARKETUTILS_HPP
#define MOR_EPETRALOCALMAPMVMATRIXMARKETUTILS_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Comm.h"

#include "Teuchos_RCP.hpp"

#include <string>

namespace MOR {

void writeLocalMapMultiVectorToMatrixMarket(
    const std::string &fileName,
    const Epetra_MultiVector &localMapMv);

Teuchos::RCP<Epetra_MultiVector> readLocalMapMultiVectorFromMatrixMarket(
    const std::string &fileName,
    const Epetra_Comm &comm,
    int vectorSize);

} // namespace MOR

#endif /* MOR_EPETRALOCALMAPMVMATRIXMARKETUTILS_HPP */
