//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedBasisFactory.hpp"

#include "Teuchos_Ptr.hpp"
#include "Teuchos_TestForException.hpp"

namespace MOR {

ReducedBasisFactory::ReducedBasisFactory()
{
  // Nothing to do
}

ReducedBasisElements
ReducedBasisFactory::create(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::Ptr<std::string> sourceId(params->getPtr<std::string>("Basis Source Type"));

  TEUCHOS_TEST_FOR_EXCEPTION(
      Teuchos::is_null(sourceId),
      std::invalid_argument,
      "Must provide a basis source"
      );

  const BasisSourceMap::const_iterator it = sources_.find(*sourceId);
  const bool sourceTypeNotFound = (it == sources_.end());

  TEUCHOS_TEST_FOR_EXCEPTION(
      sourceTypeNotFound,
      std::invalid_argument,
      sourceId << " is not a valid basis source"
      );

  return (*it->second)(params);
}

void
ReducedBasisFactory::extend(const std::string &id, const Teuchos::RCP<ReducedBasisSource> &source)
{
  sources_[id] = source;
}

} // namespace MOR
