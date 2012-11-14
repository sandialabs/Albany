//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_BASISINPUTFILE_HPP
#define ALBANY_BASISINPUTFILE_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_MultiVector;
class Epetra_Map;

namespace Albany {

Teuchos::RCP<Teuchos::ParameterList> fillDefaultBasisInputParams(
    const Teuchos::RCP<Teuchos::ParameterList> &params);

Teuchos::RCP<Epetra_MultiVector> readOrthonormalBasis(
    const Epetra_Map &basisMap,
    const Teuchos::RCP<Teuchos::ParameterList> &fileParams);

} // namespace Albany

#endif /* ALBANY_BASISINPUTFILE_HPP */
