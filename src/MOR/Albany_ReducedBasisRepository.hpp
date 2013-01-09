//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_REDUCEDBASISREPOSITORY_HPP
#define ALBANY_REDUCEDBASISREPOSITORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

class Epetra_MultiVector;

namespace Albany {

class ReducedBasisFactory;

class ReducedBasisRepository {
public:
  explicit ReducedBasisRepository(const Teuchos::RCP<ReducedBasisFactory> &factory);

  Teuchos::RCP<const Epetra_MultiVector> get(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<ReducedBasisFactory> factory_;
  typedef std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > InstanceMap;
  std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > instances_;
};

} // end namepsace Albany

#endif /* ALBANY_REDUCEDBASISREPOSITORY_HPP */


