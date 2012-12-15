//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_BASISINPUTFILE_HPP
#define ALBANY_BASISINPUTFILE_HPP

#include "Albany_LinearReducedSpaceFactory.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"

class Epetra_MultiVector;

namespace Albany {

class BasisInputFile : public LinearReducedSpaceFactory::BasisProvider {
public:
  explicit BasisInputFile(const Epetra_Map &basisMap);

  virtual Teuchos::RCP<Epetra_MultiVector> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Epetra_Map basisMap_;
};

} // namespace Albany

#endif /* ALBANY_BASISINPUTFILE_HPP */
