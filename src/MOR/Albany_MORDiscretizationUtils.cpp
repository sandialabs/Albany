//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MORDiscretizationUtils.hpp"

#include "Albany_ProblemFactory.hpp"
#include "Albany_AbstractProblem.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

void
setupInternalMeshStruct(
    DiscretizationFactory &discFactory,
    const Teuchos::RCP<Teuchos::ParameterList> &topLevelParams,
    const Teuchos::RCP<const Teuchos_Comm> &comm)
{
  const Teuchos::RCP<ParamLib> paramLib(new ParamLib);
  ProblemFactory problemFactory(topLevelParams, paramLib, comm);
  const Teuchos::RCP<AbstractProblem> problem = problemFactory.create();
  const bool sublistMustExist = true;
  const Teuchos::RCP<Teuchos::ParameterList> problemParams =
      Teuchos::sublist(topLevelParams, "Problem", sublistMustExist);
  problemParams->validateParameters(*problem->getValidProblemParameters(), 0);

  StateManager stateMgr;
  const Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > meshSpecs =
    discFactory.createMeshSpecs();
  problem->buildProblem(meshSpecs, stateMgr);

  discFactory.setupInternalMeshStruct(
      problem->numEquations(),
      stateMgr.getStateInfoStruct(),
      problem->getFieldRequirements());
}

Teuchos::RCP<AbstractDiscretization>
createDiscretization(DiscretizationFactory &discFactory)
{
  return discFactory.createDiscretizationFromInternalMeshStruct(Teuchos::null);
}

Teuchos::RCP<AbstractDiscretization>
discretizationNew(
    const Teuchos::RCP<Teuchos::ParameterList> &topLevelParams,
    const Teuchos::RCP<const Teuchos_Comm> &comm)
{
  DiscretizationFactory discFactory(topLevelParams, comm);
  setupInternalMeshStruct(discFactory, topLevelParams, comm);
  return createDiscretization(discFactory);
}

Teuchos::RCP<AbstractDiscretization>
modifiedDiscretizationNew(
    const Teuchos::RCP<Teuchos::ParameterList> &topLevelParams,
    const Teuchos::RCP<const Teuchos_Comm> &comm,
    DiscretizationTransformation &transformation)
{
  DiscretizationFactory discFactory(topLevelParams, comm);
  setupInternalMeshStruct(discFactory, topLevelParams, comm);

  transformation(discFactory);

  return createDiscretization(discFactory);
}

} // namespace Albany
