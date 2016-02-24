//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedBasisFactory.hpp"

#include "MOR_EpetraUtils.hpp"

#include "Teuchos_Ptr.hpp"
#include "Teuchos_TestForException.hpp"

namespace MOR {

namespace Detail {

ReducedBasisElements
preprocessedOrigin(const ReducedBasisElements &source, const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string type = params->get("Origin", "Default");

  if (type == "Default") {
    return source;
  } else if (type == "Zero") {
    return ReducedBasisElements(source.basis);
  } else if (type == "First Basis Vector") {
    return ReducedBasisElements(nonConstHeadView(source.basis), nonConstTailView(source.basis));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::invalid_argument,
      type << " is not a valid origin type."
      );
  return source; // Should not be reached
}

} // namespace Detail

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
      "Must provide a basis source."
      );

  const BasisSourceMap::const_iterator it = sources_.find(*sourceId);

  TEUCHOS_TEST_FOR_EXCEPTION(
      it == sources_.end(),
      std::invalid_argument,
      sourceId << " is not a valid basis source."
      );

  return Detail::preprocessedOrigin((*it->second)(params), params);
}

void
ReducedBasisFactory::extend(const std::string &id, const Teuchos::RCP<ReducedBasisSource> &source)
{
  sources_[id] = source;
}

} // namespace MOR
