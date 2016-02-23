//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedBasisRepository.hpp"

namespace MOR {

ReducedBasisRepository::ReducedBasisRepository(const Teuchos::RCP<ReducedBasisFactory> &factory) :
  factory_(factory)
{
  // Nothing to do
}

Teuchos::RCP<const Epetra_Vector>
ReducedBasisRepository::getOrigin(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const ReducedBasisElements instance = this->getInstance(params);
  return instance.origin;
}

Teuchos::RCP<const Epetra_MultiVector>
ReducedBasisRepository::getBasis(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const ReducedBasisElements instance = this->getInstance(params);
  return instance.basis;
}

ReducedBasisElements
ReducedBasisRepository::getInstance(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string key = params->name();
  const InstanceMap::iterator it = instances_.lower_bound(key);
  if (it != instances_.end() && it->first == key) {
    return it->second;
  }
  const ReducedBasisElements newInstance = factory_->create(params);
  instances_.insert(it, std::make_pair(key, newInstance));
  return newInstance;
}

} // namespace MOR
