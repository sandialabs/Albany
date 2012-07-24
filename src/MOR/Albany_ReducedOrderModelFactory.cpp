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

#include "Albany_EpetraSamplingOperator.hpp"
#include "Albany_BasisInputFile.hpp"
#include "Albany_MORUtils.hpp"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_XmlParameterListHelpers.hpp"

#include <string>
#include <stdexcept>

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;
using ::Teuchos::Array;
using ::Teuchos::Tuple;
using ::Teuchos::tuple;

Array<int> getSampleDofs(const RCP<ParameterList> &collocationParams, int dofCount) {
  const Tuple<std::string, 3> allowedSourceTypes = tuple<std::string>("All", "Inline", "File");
  const std::string sourceType = collocationParams->get("Source Type", allowedSourceTypes[0]);
  TEUCHOS_TEST_FOR_EXCEPT(!contains(allowedSourceTypes, sourceType));

  Array<int> result;
  if (sourceType == allowedSourceTypes[0]) {
    result.reserve(dofCount);
    for (int i = 0; i < dofCount; ++i) {
      result.push_back(i);
    }
  } else if (sourceType == allowedSourceTypes[1] || sourceType == allowedSourceTypes[2]) {
    RCP<ParameterList> sampleDofsParams = collocationParams;
    if (sourceType == allowedSourceTypes[2]) {
      const std::string path = Teuchos::getParameter<std::string>(*collocationParams, "Sample Dof Input File Name");
      sampleDofsParams = Teuchos::getParametersFromXmlFile(path);
    }
    result = Teuchos::getParameter<Array<int> >(*sampleDofsParams, "Sample Dof List");
  } else {
    TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Should not happen");
  }
  return result;
}

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

RCP<EpetraExt::ModelEvaluator> ReducedOrderModelFactory::create(const RCP<EpetraExt::ModelEvaluator> &child)
{
  RCP<EpetraExt::ModelEvaluator> result = child;

  if (useReducedOrderModel()) {
    const RCP<ParameterList> romParams = extractReducedOrderModelParams(params_);
    const RCP<ParameterList> fileParams = fillDefaultBasisInputParams(romParams);

    const RCP<const Epetra_Map> stateMap = child->get_x_map();
    const RCP<const Epetra_MultiVector> basis = readOrthonormalBasis(*stateMap, fileParams);
    const RCP<const ReducedSpace> reducedSpace(new LinearReducedSpace(*basis));
    RCP<const Epetra_MultiVector> leftBasis = basis;

    const Tuple<std::string, 2> allowedProjectionTypes = tuple<std::string>("Galerkin Projection", "Minimum Residual");
    const std::string projectionType = romParams->get("System Reduction", allowedProjectionTypes[0]);
    TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedProjectionTypes, projectionType),
                               std::out_of_range,
                               projectionType + " not in " + allowedProjectionTypes.toString());

    const RCP<ParameterList> hyperreductionParams = sublist(romParams, "Hyper Reduction");
    const bool useHyperreduction = hyperreductionParams->get("Activate", false);
    if (useHyperreduction) {
      const Tuple<std::string, 1> allowedHyperreductionTypes = tuple<std::string>("Collocation");
      const std::string hyperreductionType = hyperreductionParams->get("Type", allowedHyperreductionTypes[0]);
      TEUCHOS_TEST_FOR_EXCEPTION(!contains(allowedHyperreductionTypes, hyperreductionType),
                                 std::out_of_range,
                                 hyperreductionType + " not in " + allowedHyperreductionTypes.toString());
      if (hyperreductionType == allowedHyperreductionTypes[0]) {
        const Array<int> sampleDofs = getSampleDofs(sublist(hyperreductionParams, "Collocation Data"), stateMap->NumGlobalElements());
        const RCP<Epetra_MultiVector> clonedLeftBasis(new Epetra_MultiVector(leftBasis->Map(), leftBasis->NumVectors(), false));
        const EpetraSamplingOperator sampling(*stateMap, sampleDofs);
        sampling.Apply(*leftBasis, *clonedLeftBasis);
        leftBasis = clonedLeftBasis;
      }
    }

    if (projectionType == allowedProjectionTypes[0]) {
      const RCP<ReducedOperatorFactory> opFactory(new PetrovGalerkinOperatorFactory(basis, leftBasis));
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

} // end namespace Albany
