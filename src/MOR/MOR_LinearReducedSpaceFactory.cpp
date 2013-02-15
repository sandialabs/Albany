//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_LinearReducedSpaceFactory.hpp"

#include "MOR_ReducedSpace.hpp"

#include "MOR_SampleDofListFactory.hpp"
#include "MOR_EpetraSamplingOperator.hpp"
#include "MOR_ContainerUtils.hpp"
#include "MOR_EpetraUtils.hpp"
#include "MOR_BasisOps.hpp"

#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"
#include "Epetra_Map.h"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"

#include <stdexcept>

namespace MOR {

LinearReducedSpaceFactory::LinearReducedSpaceFactory(
    const Teuchos::RCP<ReducedBasisFactory> &basisFactory,
    const Teuchos::RCP<SampleDofListFactory> &samplingFactory) :
  basisRepository_(basisFactory),
  samplingFactory_(samplingFactory)
{
  // Nothing to do
}

Teuchos::RCP<LinearReducedSpace>
LinearReducedSpaceFactory::create(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<const Epetra_MultiVector> basis = this->getBasis(params);
  if (Teuchos::is_null(basis)) {
    return Teuchos::null;
  }

  const Teuchos::RCP<const Epetra_MultiVector> projector = this->getProjector(params);

  return Teuchos::rcp(new LinearReducedSpace(basis, projector));
}

Teuchos::RCP<const Epetra_MultiVector>
LinearReducedSpaceFactory::getBasis(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  return basisRepository_.get(params);
}

Teuchos::RCP<const Epetra_MultiVector>
LinearReducedSpaceFactory::getProjector(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<const Epetra_MultiVector> basis = this->getBasis(params);

  const Teuchos::RCP<const Epetra_Map> basisMap = mapDowncast(basis->Map());
  TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(basisMap));

  const Teuchos::RCP<const Epetra_Operator> collocationOperator =
    this->getSamplingOperator(params, *basisMap);

  Teuchos::RCP<const Epetra_MultiVector> result = basis;
  if (Teuchos::nonnull(collocationOperator)) {
    const Teuchos::RCP<Epetra_MultiVector> dualBasis(
        new Epetra_MultiVector(collocationOperator->OperatorRangeMap(), basis->NumVectors(), false));
    dualize(*basis, *collocationOperator, *dualBasis);
    result = dualBasis;
  }
  return result;
}

Teuchos::RCP<const Epetra_Operator>
LinearReducedSpaceFactory::getSamplingOperator(
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    const Epetra_Map &stateMap)
{
  Teuchos::RCP<const Epetra_Operator> result;
  {
    const Teuchos::RCP<Teuchos::ParameterList> hyperreductionParams = Teuchos::sublist(params, "Hyper Reduction");
    const bool useHyperreduction = hyperreductionParams->get("Activate", false);
    if (useHyperreduction) {
      const Teuchos::Tuple<std::string, 1> allowedHyperreductionTypes = Teuchos::tuple<std::string>("Collocation");
      const std::string hyperreductionType = hyperreductionParams->get("Type", allowedHyperreductionTypes[0]);
      TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedHyperreductionTypes, hyperreductionType),
          std::out_of_range,
          hyperreductionType + " not in " + allowedHyperreductionTypes.toString());
      if (hyperreductionType == allowedHyperreductionTypes[0]) {
        const Teuchos::RCP<Teuchos::ParameterList> collocationParams = Teuchos::sublist(hyperreductionParams, "Collocation Data");
        const Teuchos::Array<int> sampleLocalEntries = samplingFactory_->create(collocationParams);
        result = Teuchos::rcp(new EpetraSamplingOperator(stateMap, sampleLocalEntries));
      } else {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Should not happen");
      }
    }
  }
  return result;
}

} // namespace MOR
