//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_BASISINPUTFILE_HPP
#define MOR_BASISINPUTFILE_HPP

#include "MOR_ReducedBasisFactory.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_Map.h"

class Epetra_MultiVector;

namespace MOR {

class BasisInputFile : public ReducedBasisFactory::BasisProvider {
public:
  explicit BasisInputFile(const Epetra_Map &basisMap);

  virtual Teuchos::RCP<Epetra_MultiVector> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Epetra_Map basisMap_;
};

} // namespace MOR

#endif /* MOR_BASISINPUTFILE_HPP */
