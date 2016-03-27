//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_SampleDofListFactory.hpp"

#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace MOR {

SampleDofListFactory::SampleDofListFactory()
{
  // Nothing to do
}

Teuchos::Array<int>
SampleDofListFactory::create(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string providerId = params->get("Source Type", "");

  const ProviderMap::const_iterator it = providers_.find(providerId);
  TEUCHOS_TEST_FOR_EXCEPTION(
      it == providers_.end(),
      std::out_of_range,
      "Unknown sample dof provider type: " << providerId);

  return (*it->second)(params);
}

void
SampleDofListFactory::extend(const std::string &id, const Teuchos::RCP<DofListProvider> &provider)
{
  providers_[id] = provider;
}

} // namespace MOR
