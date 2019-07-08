//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //


//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EIGEN_DATA_STRUCT_HPP
#define ALBANY_EIGEN_DATA_STRUCT_HPP

#include <string>
#include <vector>

#include "Albany_ThyraTypes.hpp"

#include <Teuchos_RCP.hpp>

namespace Albany {

struct EigendataStruct {

  EigendataStruct () = default;
  ~EigendataStruct () = default;

  EigendataStruct (const EigendataStruct& src) {
    eigenvalueRe = Teuchos::rcp(new std::vector<double>(*(src.eigenvalueRe)));
    eigenvalueIm = Teuchos::rcp(new std::vector<double>(*(src.eigenvalueIm)));

    eigenvectorRe = Thyra::createMembers(src.eigenvectorRe->range(),src.eigenvectorRe->domain()->dim());
    eigenvectorIm = Thyra::createMembers(src.eigenvectorIm->range(),src.eigenvectorIm->domain()->dim());
  };

  Teuchos::RCP<std::vector<double> >  eigenvalueRe;
  Teuchos::RCP<std::vector<double> >  eigenvalueIm;
  Teuchos::RCP<Thyra_MultiVector>     eigenvectorRe;
  Teuchos::RCP<Thyra_MultiVector>     eigenvectorIm;
};

} // namespace Albany

#endif // ALBANY_EIGEN_DATA_STRUCT_HPP
