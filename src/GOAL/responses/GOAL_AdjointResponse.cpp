//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponse.hpp"
#include "GOAL_MechanicsProblem.hpp"

namespace GOAL {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;

using Albany::Application;
using Albany::AbstractProblem;
using Albany::StateManager;
using Albany::MeshSpecsStruct;

AdjointResponse::AdjointResponse(
    const RCP<Application>& app,
    const RCP<AbstractProblem>& prob,
    const RCP<StateManager>& sm,
    const ArrayRCP<RCP<MeshSpecsStruct> >& ms,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm())
{
  RCP<Albany::GOALMechanicsProblem> problem = 
    Teuchos::rcp_dynamic_cast<Albany::GOALMechanicsProblem>(prob);
  problem->buildAdjointProblem(ms, *sm, rcp(&rp, false));
}

AdjointResponse::~AdjointResponse()
{
}

void AdjointResponse::evaluateResponseT(
    const double currentTime,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
}

}
