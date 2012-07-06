/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_ReducedOrderModelFactory.hpp"

#include "Albany_ReducedOrderModelEvaluator.hpp"

#include "Albany_PetrovGalerkinOperatorFactory.hpp"
#include "Albany_GaussNewtonOperatorFactory.hpp"

#include "Albany_ReducedSpace.hpp"
#include "Albany_MultiVectorInputFile.hpp"
#include "Albany_MultiVectorInputFileFactory.hpp"
#include "Albany_MultiVectorOutputFile.hpp"
#include "Albany_MultiVectorOutputFileFactory.hpp"

#include "Albany_MORUtils.hpp"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_TestForException.hpp"

#include <string>

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;
using ::Teuchos::Tuple;
using ::Teuchos::tuple;

ReducedOrderModelFactory::ReducedOrderModelFactory(const RCP<ParameterList> &parentParams) :
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

RCP<ParameterList> ReducedOrderModelFactory::fillDefaultReducedOrderModelParams(const RCP<ParameterList> &params)
{
  params->get("Input File Group Name", "basis");
  params->get("Input File Default Base File Name", "basis");
  return params;
}

RCP<EpetraExt::ModelEvaluator> ReducedOrderModelFactory::create(const RCP<EpetraExt::ModelEvaluator> &child)
{
  RCP<EpetraExt::ModelEvaluator> result = child;

  if (useReducedOrderModel()) {
    const RCP<ParameterList> romParams = extractReducedOrderModelParams(params_);
    const RCP<const Epetra_MultiVector> basis = createOrthonormalBasis(*child->get_x_map(),
                                                                       romParams);
    const RCP<const ReducedSpace> reducedSpace(new LinearReducedSpace(*basis));

    static const Tuple<std::string, 2> allowedProjectionTypes = tuple<std::string>("Galerkin Projection", "Minimum Residual");
    const std::string projectionType = romParams->get("Equation Reduction", allowedProjectionTypes[0]);
    TEUCHOS_TEST_FOR_EXCEPT(!contains(allowedProjectionTypes, projectionType));

    if (projectionType == allowedProjectionTypes[0]) {
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis));
      result = rcp(new ReducedOrderModelEvaluator(child, reducedSpace, opFactory));
    } else if (projectionType == allowedProjectionTypes[1]) {
      const RCP<ReducedOperatorFactory> opFactory(new GaussNewtonOperatorFactory(basis));
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

RCP<Epetra_MultiVector> ReducedOrderModelFactory::createOrthonormalBasis(const Epetra_Map &fullStateMap,
                                                                         const RCP<ParameterList> &params)
{
  const RCP<ParameterList> fileParams = fillDefaultReducedOrderModelParams(params);
  MultiVectorInputFileFactory factory(fileParams);

  // TODO read partial basis
  const RCP<MultiVectorInputFile> file = factory.create();
  return file->read(fullStateMap);
}

} // end namespace Albany
