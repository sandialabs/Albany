//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedOrderModelFactory.hpp"

#include "MOR_ReducedSpaceFactory.hpp"
#include "MOR_ReducedSpace.hpp"

#include "MOR_BasisOps.hpp"

#include "MOR_SampleDofListFactory.hpp"

#include "MOR_ReducedOrderModelEvaluator.hpp"
#include "MOR_PetrovGalerkinOperatorFactory.hpp"
#include "MOR_GaussNewtonOperatorFactory.hpp"

#include "MOR_ContainerUtils.hpp"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_TestForException.hpp"

#include <string>
#include <stdexcept>

namespace MOR {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::nonnull;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;
using ::Teuchos::Tuple;
using ::Teuchos::tuple;

ReducedOrderModelFactory::ReducedOrderModelFactory(
    const Teuchos::RCP<ReducedSpaceFactory> &spaceFactory,
    const RCP<ParameterList> &parentParams) :
  spaceFactory_(spaceFactory),
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

    const Tuple<std::string, 2> allowedProjectionTypes = tuple<std::string>("Galerkin Projection", "Minimum Residual");
    const std::string projectionType = romParams->get("System Reduction", allowedProjectionTypes[0]);
    TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedProjectionTypes, projectionType),
                               std::out_of_range,
                               projectionType + " not in " + allowedProjectionTypes.toString());

    const RCP<const ReducedSpace> reducedSpace = spaceFactory_->create(romParams);
    const RCP<const Epetra_MultiVector> basis = spaceFactory_->getBasis(romParams);

    if (projectionType == allowedProjectionTypes[0]) {
      const RCP<const Epetra_MultiVector> projector = spaceFactory_->getProjector(romParams);
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis, projector));
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory));
    } else if (projectionType == allowedProjectionTypes[1]) {
      RCP<ReducedOperatorFactory> opFactory;

      const RCP<const Epetra_Operator> collocationOperator =
        spaceFactory_->getSamplingOperator(romParams, *child->get_x_map());
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

} // namespace MOR
