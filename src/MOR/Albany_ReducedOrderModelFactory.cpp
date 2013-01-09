//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ReducedOrderModelFactory.hpp"

#include "Albany_LinearReducedSpaceFactory.hpp"
#include "Albany_ReducedSpace.hpp"

#include "Albany_SampleDofListFactory.hpp"

#include "Albany_ReducedOrderModelEvaluator.hpp"
#include "Albany_PetrovGalerkinOperatorFactory.hpp"
#include "Albany_GaussNewtonOperatorFactory.hpp"

#include "Albany_EpetraSamplingOperator.hpp"
#include "Albany_MORUtils.hpp"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include <string>
#include <stdexcept>

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::nonnull;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;
using ::Teuchos::Array;
using ::Teuchos::Tuple;
using ::Teuchos::tuple;

ReducedOrderModelFactory::ReducedOrderModelFactory(
    const Teuchos::RCP<LinearReducedSpaceFactory> &spaceFactory,
    const Teuchos::RCP<SampleDofListFactory> &samplingFactory,
    const RCP<ParameterList> &parentParams) :
  spaceFactory_(spaceFactory),
  samplingFactory_(samplingFactory),
  params_(extractModelOrderReductionParams(parentParams))
{
  // Nothing to do
}

RCP<ParameterList> ReducedOrderModelFactory::extractModelOrderReductionParams(const RCP<ParameterList> &params)
{
  return sublist(params, "Model Order Reduction");
}

RCP<ParameterList> ReducedOrderModelFactory::extractReducedOrderModelParams(const RCP<ParameterList> &params)
{
  return sublist(params, "Reduced-Order Model");
}

RCP<EpetraExt::ModelEvaluator> ReducedOrderModelFactory::create(const RCP<EpetraExt::ModelEvaluator> &child)
{
  RCP<EpetraExt::ModelEvaluator> result = child;

  if (useReducedOrderModel()) {
    const RCP<ParameterList> romParams = extractReducedOrderModelParams(params_);

    const RCP<const Epetra_Map> stateMap = child->get_x_map();

    const RCP<ParameterList> fileParams = romParams;
    const RCP<const ReducedSpace> reducedSpace = spaceFactory_->create(fileParams);
    const RCP<const Epetra_MultiVector> basis = spaceFactory_->getBasis(fileParams);

    const Tuple<std::string, 2> allowedProjectionTypes = tuple<std::string>("Galerkin Projection", "Minimum Residual");
    const std::string projectionType = romParams->get("System Reduction", allowedProjectionTypes[0]);
    TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedProjectionTypes, projectionType),
                               std::out_of_range,
                               projectionType + " not in " + allowedProjectionTypes.toString());

    RCP<const Epetra_Operator> collocationOperator;
    {
      const RCP<ParameterList> hyperreductionParams = sublist(romParams, "Hyper Reduction");
      const bool useHyperreduction = hyperreductionParams->get("Activate", false);
      if (useHyperreduction) {
        const Tuple<std::string, 1> allowedHyperreductionTypes = tuple<std::string>("Collocation");
        const std::string hyperreductionType = hyperreductionParams->get("Type", allowedHyperreductionTypes[0]);
        TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedHyperreductionTypes, hyperreductionType),
            std::out_of_range,
            hyperreductionType + " not in " + allowedHyperreductionTypes.toString());
        if (hyperreductionType == allowedHyperreductionTypes[0]) {
          const RCP<ParameterList> collocationParams = sublist(hyperreductionParams, "Collocation Data");
          const Array<int> sampleLocalEntries = samplingFactory_->create(collocationParams);
          collocationOperator = rcp(new EpetraSamplingOperator(*stateMap, sampleLocalEntries));
        } else {
          TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Should not happen");
        }
      }
    }

    if (projectionType == allowedProjectionTypes[0]) {
      RCP<const Epetra_MultiVector> leftBasis = basis;
      if (nonnull(collocationOperator)) {
        const RCP<Epetra_MultiVector> clonedLeftBasis(new Epetra_MultiVector(leftBasis->Map(), leftBasis->NumVectors(), false));
        collocationOperator->Apply(*leftBasis, *clonedLeftBasis);
        leftBasis = clonedLeftBasis;
      }
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis, leftBasis));
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory));
    } else if (projectionType == allowedProjectionTypes[1]) {
      RCP<ReducedOperatorFactory> opFactory;
      if (nonnull(collocationOperator)) {
        opFactory = rcp(new GaussNewtonMetricOperatorFactory(basis, collocationOperator));
      } else {
        opFactory = rcp(new GaussNewtonOperatorFactory(basis));
      }
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory));
    } else {
      TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Should not happen");
    }
  }

  return result;
}

bool ReducedOrderModelFactory::useReducedOrderModel() const
{
  return extractReducedOrderModelParams(params_)->get("Activate", false);
}

} // end namespace Albany
