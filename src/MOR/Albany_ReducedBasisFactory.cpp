//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ReducedBasisFactory.hpp"

#include "Epetra_MultiVector.h"

namespace Albany {

ReducedBasisFactory::ReducedBasisFactory()
{
  // Nothing to do
}

Teuchos::RCP<Epetra_MultiVector>
ReducedBasisFactory::create(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string providerId = params->get("Basis Source Type", "");

  Teuchos::RCP<Epetra_MultiVector> result;

  if (!providerId.empty()) {
    const BasisProviderMap::const_iterator it = mvProviders_.find(providerId);
    if (it != mvProviders_.end()) {
      result = (*it->second)(params);
    }
  }

  return result;
}

void
ReducedBasisFactory::extend(const std::string &id, const Teuchos::RCP<BasisProvider> &provider)
{
  mvProviders_[id] = provider;
}

} // end namespace Albany

