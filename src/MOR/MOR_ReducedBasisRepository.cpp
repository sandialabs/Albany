//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedBasisRepository.hpp"

#include "MOR_ReducedBasisFactory.hpp"

#include "Epetra_MultiVector.h"

namespace MOR {

ReducedBasisRepository::ReducedBasisRepository(const Teuchos::RCP<ReducedBasisFactory> &factory) :
  factory_(factory)
{
  // Nothing to do
}

Teuchos::RCP<const Epetra_MultiVector>
ReducedBasisRepository::get(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string key = params->name();
  const InstanceMap::iterator it = instances_.lower_bound(key);
  if (it != instances_.end() && it->first == key) {
    return it->second;
  }
  const ReducedBasisElements newInstance = factory_->create(params);
  const Teuchos::RCP<Epetra_MultiVector> newBasis = newInstance.basis;
  instances_.insert(it, std::make_pair(key, newBasis));
  return newBasis;
}

} // namespace MOR
