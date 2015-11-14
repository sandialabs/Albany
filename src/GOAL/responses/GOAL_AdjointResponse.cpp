//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponse.hpp"

namespace GOAL {

using Teuchos::RCP;
using Teuchos::ArrayRCP;

AdjointResponse::AdjointResponse(
    RCP<Albany::Application> const& app,
    RCP<Albany::AbstractProblem> const& prob,
    RCP<Albany::StateManager> const& sm,
    ArrayRCP<RCP<Albany::MeshSpecsStruct> > const& ms,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm())
{
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
