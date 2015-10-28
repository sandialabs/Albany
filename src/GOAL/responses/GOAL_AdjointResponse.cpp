//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponse.hpp"

namespace GOAL {

MechAdjResponse::MechAdjResponse(
    const RCP<Application>& app,
    const RCP<AbstractProblem>& prob,
    const RCP<StateManager>& sm,
    const ArrayRCP<RCP<MeshSpecsStruct> >& ms,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm()),
{
}

MechAdjResponse::~MechAdjResponse()
{
}

void MechAdjResponse::evaluateResponseT(
    const double currentTime,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
}

}
