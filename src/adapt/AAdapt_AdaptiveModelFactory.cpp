//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "AAdapt_AdaptiveModelFactory.hpp"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_TestForException.hpp"

#include <string>
#include <stdexcept>

namespace AAdapt {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::nonnull;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;
using ::Teuchos::Tuple;
using ::Teuchos::tuple;

AdaptiveModelFactory::AdaptiveModelFactory(
  const RCP<ParameterList>& parentParams) :
  params(extractAdaptiveModelParams(parentParams)) {
  // Nothing to do
}

RCP<ParameterList> AdaptiveModelFactory::extractAdaptiveModelParams(const RCP<ParameterList>& params_) {

  const Teuchos::RCP<Teuchos::ParameterList>& problemParams =
    Teuchos::sublist(params_, "Problem", true);

  if(problemParams->isSublist("Adaptation")) { // If the user has specified adaptation on input, grab the sublist

    return sublist(problemParams, "Adaptation");

  }

  return Teuchos::null;

}

RCP<EpetraExt::ModelEvaluator> AdaptiveModelFactory::create(const RCP<EpetraExt::ModelEvaluator>& child) {
  RCP<EpetraExt::ModelEvaluator> result = child;

  if(useAdaptiveModel()) {
#if 0
    const RCP<ParameterList> romParams = extractAdaptiveModelParams(params_);

    const Tuple<std::string, 2> allowedProjectionTypes = tuple<std::string>("Galerkin Projection", "Minimum Residual");
    const std::string projectionType = romParams->get("System Reduction", allowedProjectionTypes[0]);
    TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedProjectionTypes, projectionType),
                               std::out_of_range,
                               projectionType + " not in " + allowedProjectionTypes.toString());

    const RCP<const ReducedSpace> reducedSpace = spaceFactory_->create(romParams);
    const RCP<const Epetra_MultiVector> basis = spaceFactory_->getBasis(romParams);

    if(projectionType == allowedProjectionTypes[0]) {
      const RCP<const Epetra_MultiVector> projector = spaceFactory_->getProjector(romParams);
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis, projector));
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory));
    }

    else if(projectionType == allowedProjectionTypes[1]) {
      RCP<ReducedOperatorFactory> opFactory;

      const RCP<const Epetra_Operator> collocationOperator =
        spaceFactory_->getSamplingOperator(romParams, *child->get_x_map());

      if(nonnull(collocationOperator)) {
        opFactory = rcp(new GaussNewtonMetricOperatorFactory(basis, collocationOperator));
      }

      else {
        opFactory = rcp(new GaussNewtonOperatorFactory(basis));
      }

      result = rcp(new AdaptiveModelEvaluator(child, reducedSpace, opFactory));
    }

    else {
      TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Should not happen");
    }

#endif
  }

  return result;
}

bool AdaptiveModelFactory::useAdaptiveModel() const {

  const Teuchos::RCP<Teuchos::ParameterList>& problemParams =
    Teuchos::sublist(params, "Problem", true);

  if(problemParams->isSublist("Adaptation")) { // If the user has specified adaptation on input, grab the sublist

    return true;

  }

  return false;

}

} // namespace AAdapt
