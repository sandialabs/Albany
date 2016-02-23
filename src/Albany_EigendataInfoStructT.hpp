//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //


//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_EIGENDATASTRUCTT
#define ALBANY_EIGENDATASTRUCTT

#include <string>
#include <vector>
#include "Teuchos_RCP.hpp"
#include "Tpetra_Vector.hpp"
#include "Albany_DataTypes.hpp"

namespace Albany {

struct EigendataStructT {

  EigendataStructT () {};
  ~EigendataStructT () {};
  EigendataStructT (const EigendataStructT& copy) {
    eigenvalueRe = Teuchos::rcp(new std::vector<double>(*(copy.eigenvalueRe)));
    eigenvalueIm = Teuchos::rcp(new std::vector<double>(*(copy.eigenvalueIm)));
    eigenvectorRe = Teuchos::rcp(new Tpetra_MultiVector(*(copy.eigenvectorRe)));
    eigenvectorIm = Teuchos::rcp(new Tpetra_MultiVector(*(copy.eigenvectorIm)));
  };

  Teuchos::RCP<std::vector<double> > eigenvalueRe;
  Teuchos::RCP<std::vector<double> > eigenvalueIm;
  Teuchos::RCP<Tpetra_MultiVector> eigenvectorRe;
  Teuchos::RCP<Tpetra_MultiVector> eigenvectorIm;
};

}
#endif



