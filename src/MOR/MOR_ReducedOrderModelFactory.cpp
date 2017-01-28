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

    const int num_DBC_modes = romParams->get("Number of DBC Modes", 0);
    printf("Parameter read: num_DBC_modes = %d.\n", num_DBC_modes);
    if (projectionType == allowedProjectionTypes[0])
    {
      if (num_DBC_modes != 0)
      {
        printf("WARNING:  Galerkin Projection selected, specifying Number of DBC Modes will have no effect.\n");
      }
    }
    else if (projectionType == allowedProjectionTypes[1])
    {
      printf("Minimum Residual ROM will be run with %d DBC Modes.\n", num_DBC_modes);
    }
    else
    {
      if (num_DBC_modes != 0)
      {
        printf("WARNING:  Unknown projection type, specifying Number of DBC Modes will have no effect.\n");
      }
    }

    std::string preconditionerType = romParams->get("Preconditioner Type", "None");
    printf("Parameter read: preconditionerType = %s.\n", preconditionerType.c_str());
    if (projectionType == allowedProjectionTypes[0])
    {
      if (preconditionerType.compare("None") != 0)
      {
        printf("WARNING:  Galerkin Projection selected, preconditioning is not supported, setting preconditionerType to None.\n");
        preconditionerType = "None";
      }
    }
    else if (projectionType == allowedProjectionTypes[1])
    {
      printf("Minimum Residual ROM will be run with preconditioner = %s.\n", preconditionerType.c_str());
    }
    else
    {
      if (preconditionerType.compare("None") != 0)
      {
        printf("WARNING:  Unknown projection type selected, preconditioning is not supported, setting preconditionerType to None.\n");
        preconditionerType = "None";
      }
    }


    const bool outputTrace = romParams->get("Output Trace", false);
    printf("Parameter read: outputTrace = %d.\n", outputTrace);

    const bool writeJacobian = romParams->get("Write Jacobian to File", false);
    printf("Parameter read: writeJacobian = %d.\n", writeJacobian);

    const bool writeResidual = romParams->get("Write Residual to File", false);
    printf("Parameter read: writeResidual = %d.\n", writeResidual);

    const bool writeSolution = romParams->get("Write Solution to File", false);
    printf("Parameter read: writeSolution = %d.\n", writeSolution);

    const bool writePreconditioner = romParams->get("Write Preconditioner to File", false);
    printf("Parameter read: writePreconditioner = %d.\n", writePreconditioner);

    bool output_flags[8];
    output_flags[0] = outputTrace;
    output_flags[1] = writeJacobian;
    output_flags[2] = writeResidual;
    output_flags[3] = writeSolution;
    output_flags[4] = writePreconditioner;

    const RCP<const ReducedSpace> reducedSpace = spaceFactory_->create(romParams);
    const RCP<const Epetra_MultiVector> basis = spaceFactory_->getBasis(romParams);

    if (projectionType == allowedProjectionTypes[0]) {
      const RCP<const Epetra_MultiVector> projector = spaceFactory_->getProjector(romParams);
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis, projector));
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory, output_flags, preconditionerType));
    } else if (projectionType == allowedProjectionTypes[1]) {
      RCP<ReducedOperatorFactory> opFactory;

      const RCP<const Epetra_Operator> collocationOperator =
        spaceFactory_->getSamplingOperator(romParams, *child->get_x_map());
      if (nonnull(collocationOperator)) {
        opFactory = rcp(new GaussNewtonMetricOperatorFactory(basis, collocationOperator));
      } else {
        opFactory = rcp(new GaussNewtonOperatorFactory(basis, num_DBC_modes));
      }

      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory, output_flags, preconditionerType));
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
