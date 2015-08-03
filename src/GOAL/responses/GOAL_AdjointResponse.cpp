//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponse.hpp"
#include "GOAL_MechanicsProblem.hpp"
#include "PHAL_Workset.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace GOAL {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::rcp_dynamic_cast;

using Albany::Application;
using Albany::AbstractProblem;
using Albany::StateManager;
using Albany::MeshSpecsStruct;

static void print(const char* msg)
{
  RCP<Teuchos::FancyOStream> out =
    Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "GOAL: " << msg << std::endl;
}

AdjointResponse::AdjointResponse(
    const RCP<Application>& app,
    const RCP<AbstractProblem>& prob,
    const RCP<StateManager>& sm,
    const ArrayRCP<RCP<MeshSpecsStruct> >& ms,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm())
{
  RCP<Albany::GOALMechanicsProblem> problem = 
    rcp_dynamic_cast<Albany::GOALMechanicsProblem>(prob);

  problem->buildAdjointProblem(ms, *sm, rcp(&rp, false));

  fm = problem->getAdjointFieldManager();
  dfm = problem->getAdjointDirichletFieldManager();
  qfm = problem->getAdjointQoIFieldManager();
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
  if (evalCtr == 0) {evalCtr++; return;}
  print("solving adjoint problem");
}

}
