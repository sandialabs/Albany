//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDBASISELEMENTS_HPP
#define MOR_REDUCEDBASISELEMENTS_HPP

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

namespace MOR {

struct ReducedBasisElements {
  /* implicit */ ReducedBasisElements(
      const Teuchos::RCP<Epetra_MultiVector> &basis_in) :
    origin(), basis(basis_in)
  {}

  ReducedBasisElements(
      const Teuchos::RCP<Epetra_Vector> &origin_in,
      const Teuchos::RCP<Epetra_MultiVector> &basis_in) :
    origin(origin_in), basis(basis_in)
  {}

  Teuchos::RCP<Epetra_Vector> origin;
  Teuchos::RCP<Epetra_MultiVector> basis;
};

} // end namepsace MOR

#endif /* MOR_REDUCEDBASISELEMENTS_HPP */

