//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDBASISFACTORY_HPP
#define MOR_REDUCEDBASISFACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

class Epetra_MultiVector;

namespace MOR {

class ReducedBasisFactory {
public:
  ReducedBasisFactory();

  Teuchos::RCP<Epetra_MultiVector> create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  class BasisProvider;
  void extend(const std::string &id, const Teuchos::RCP<BasisProvider> &provider);

private:
  typedef std::map<std::string, Teuchos::RCP<BasisProvider> > BasisProviderMap;
  BasisProviderMap mvProviders_;
};

class ReducedBasisFactory::BasisProvider {
public:
  virtual Teuchos::RCP<Epetra_MultiVector> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) = 0;
  virtual ~BasisProvider() {}
};

} // end namepsace Albany

#endif /* MOR_REDUCEDBASISFACTORY_HPP */

