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

    std::string preconditionerType = romParams->get("Preconditioner Type", "None");
    if (projectionType == allowedProjectionTypes[0])
      if (preconditionerType.compare("None") != 0)
        preconditionerType = "None";
    else
      if (preconditionerType.compare("None") != 0)
        preconditionerType = "None";


    const bool outputTrace = romParams->get("Output Trace", false);
    const bool writeJacobian = romParams->get("Write Jacobian to File", false);
    const bool writeResidual = romParams->get("Write Residual to File", false);
    const bool writeSolution = romParams->get("Write Solution to File", false);
    const bool writePreconditioner = romParams->get("Write Preconditioner to File", false);

    bool output_flags[8];
    output_flags[0] = outputTrace;
    output_flags[1] = writeJacobian;
    output_flags[2] = writeResidual;
    output_flags[3] = writeSolution;
    output_flags[4] = writePreconditioner;
	
	bool pretendWereGalerkin = (preconditionerType.compare("Mimic Galerkin") == 0);
	bool runWithQR = romParams->get<bool>("Enable QR", false);

    const RCP<const ReducedSpace> reducedSpace = spaceFactory_->create(romParams);
    const RCP<const Epetra_MultiVector> basis = spaceFactory_->getBasis(romParams);


	if (basis->Comm().MyPID() == 0)
	{
		std::cout << "Parameter read: num_DBC_modes = " <<  num_DBC_modes << std::endl;
		if (projectionType == allowedProjectionTypes[0])
		  std::cout << "Galerkin Projection ROM will be run with " << num_DBC_modes << " DBC Modes." << std::endl;
		else if (projectionType == allowedProjectionTypes[1])
		  std::cout << "Minimum Residual ROM will be run with " << num_DBC_modes << " DBC Modes." << std::endl;
		else
		  if (num_DBC_modes != 0)
		    std::cout << "WARNING:  Unknown projection type, specifying Number of DBC Modes will have no effect." << std::endl;

		std::cout << "Parameter read: preconditionerType = " <<  preconditionerType.c_str() << std::endl;
		if (projectionType == allowedProjectionTypes[0])
		  if (preconditionerType.compare("None") != 0)
		    std::cout << "WARNING:  Galerkin Projection selected, preconditioning is not supported, setting preconditionerType to None." << std::endl;
		else if (projectionType == allowedProjectionTypes[1])
		  std::cout << "Minimum Residual ROM will be run with preconditioner = " <<  preconditionerType.c_str() << std::endl;
		else
		  if (preconditionerType.compare("None") != 0)
		    std::cout << "WARNING:  Unknown projection type selected, preconditioning is not supported, setting preconditionerType to None." << std::endl;

		std::cout << "Parameter read: outputTrace = " <<  outputTrace << std::endl;
		std::cout << "Parameter read: writeJacobian = " <<  writeJacobian << std::endl;
		std::cout << "Parameter read: writeResidual = " <<  writeResidual << std::endl;
		std::cout << "Parameter read: writeSolution = " <<  writeSolution << std::endl;
		std::cout << "Parameter read: writePreconditioner = " <<  writePreconditioner << std::endl;
	}


    if (projectionType == allowedProjectionTypes[0]) {
      const RCP<const Epetra_MultiVector> projector = spaceFactory_->getProjector(romParams);
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis, projector));
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory, output_flags, preconditionerType));
    } else if (projectionType == allowedProjectionTypes[1]) {
      RCP<ReducedOperatorFactory> opFactory;

      const RCP<const Epetra_Operator> collocationOperator =
        spaceFactory_->getSamplingOperator(romParams, *child->get_x_map());
      if (nonnull(collocationOperator)) {
        opFactory = rcp(new GaussNewtonMetricOperatorFactory(basis, collocationOperator, pretendWereGalerkin, runWithQR));
      } else {
        opFactory = rcp(new GaussNewtonOperatorFactory(basis, pretendWereGalerkin, runWithQR));
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
