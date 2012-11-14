//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EIGENDATASTRUCT
#define ALBANY_EIGENDATASTRUCT

#include <string>
#include <vector>
#include "Teuchos_RCP.hpp"
#include "Epetra_Vector.h"

namespace Albany {

struct EigendataStruct {

  EigendataStruct () {};
  ~EigendataStruct () {};

  Teuchos::RCP<std::vector<double> > eigenvalueRe;
  Teuchos::RCP<std::vector<double> > eigenvalueIm;
  Teuchos::RCP<Epetra_MultiVector> eigenvectorRe;
  Teuchos::RCP<Epetra_MultiVector> eigenvectorIm;
};

}
#endif



