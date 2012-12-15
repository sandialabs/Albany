//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_LinearReducedSpaceFactory.hpp"

#include "Albany_ReducedSpace.hpp"

#include "Epetra_MultiVector.h"

namespace Albany {

LinearReducedSpaceFactory::LinearReducedSpaceFactory()
{
  // Nothing to do
}

Teuchos::RCP<LinearReducedSpace>
LinearReducedSpaceFactory::create(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string providerId = params->get("Basis Source Type", "");

  if (!providerId.empty()) {
    const BasisProviderMap::const_iterator it = mvProviders_.find(providerId);
    if (it != mvProviders_.end()) {
      const Teuchos::RCP<const Epetra_MultiVector> basis = (*it->second)(params);
      return Teuchos::rcp(new LinearReducedSpace(*basis));
    }
  }

  return Teuchos::null;
}

void
LinearReducedSpaceFactory::extend(const std::string &id, const Teuchos::RCP<BasisProvider> &provider)
{
  mvProviders_[id] = provider;
}

} // end namespace Albany
