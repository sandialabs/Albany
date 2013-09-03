//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MORDiscretizationUtils.hpp"

#include "Albany_ProblemFactory.hpp"
#include "Albany_AbstractProblem.hpp"

namespace Albany {

void
setupInternalMeshStruct(
    DiscretizationFactory &discFactory,
    const Teuchos::RCP<Teuchos::ParameterList> &problemParams,
    const Teuchos::RCP<const Epetra_Comm> &epetraComm)
{
  const Teuchos::RCP<ParamLib> paramLib(new ParamLib);
  ProblemFactory problemFactory(problemParams, paramLib, epetraComm);
  const Teuchos::RCP<AbstractProblem> problem = problemFactory.create();
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
    DiscretizationFactory &discFactory,
    const Teuchos::RCP<Teuchos::ParameterList> &problemParams,
    const Teuchos::RCP<const Epetra_Comm> &epetraComm)
{
  setupInternalMeshStruct(discFactory, problemParams, epetraComm);
  return createDiscretization(discFactory);
}

} // namespace Albany
