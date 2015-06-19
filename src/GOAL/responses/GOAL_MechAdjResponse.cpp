//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_MechAdjResponse.hpp"
#include "GOAL_BCManager.hpp"
#include "GOAL_FieldManagerBundle.hpp"
#include "GOAL_Discretization.hpp"
#include "GOAL_LinearSystem.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_APFDiscretization.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace GOAL {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::rcpFromRef;
using Albany::Application;
using Albany::AbstractProblem;
using Albany::StateManager;
using Albany::MeshSpecsStruct;
using Albany::APFDiscretization;

static RCP<Teuchos::ParameterList> getValidParams()
{
  RCP<Teuchos::ParameterList> validPL =
    rcp(new Teuchos::ParameterList("Valid MechAdjResponse Params"));
  validPL->set<std::string>("Name", "Mechanics Adjoint",
      "response function name");
  validPL->set<bool>("Enriched Solve", true,
      "solve with enriched basis functions");
  validPL->set<bool>("Write Phalanx Graphs", false,
      "write Phalanx Graph visualization files");
  validPL->set<bool>("Write Linear System", false,
      "write linear system to files");
  return validPL;
}

MechAdjResponse::MechAdjResponse(
    const RCP<Application>& app,
    const RCP<AbstractProblem>& prob,
    const RCP<StateManager>& sm,
    const ArrayRCP<RCP<MeshSpecsStruct> >& ms,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm()),
  evalCtr(0)
{
  pb = rcp( new ProblemBundle(rp,app,prob,sm,ms) );
  Teuchos::RCP<BCManager> bcm = app->getBCManager();

  RCP<const Teuchos::ParameterList> vp = getValidParams();
  pb->params.validateParameters(*vp,0);
  enrich = pb->params.get<bool>("Enriched Solve",true);
  writePHXGraphs = pb->params.get<bool>("Write Phalanx Graphs",false); 
  writeLinearSystem = pb->params.get<bool>("Write Linear System",false);
  pb->enrich = enrich;

  fmb = rcp( new FieldManagerBundle(bcm,pb) );
  if (writePHXGraphs)
    fmb->writePHXGraphs();
}

MechAdjResponse::~MechAdjResponse()
{
}

static void print(const char* msg)
{
  RCP<Teuchos::FancyOStream> out = 
    Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "GOAL: " << msg << std::endl;
}

static void setupInitialWorksetInfo(
    PHAL::Workset& workset,
    const double time)
{
  workset.is_adjoint = false;
  workset.j_coeff = 1.0;
  workset.m_coeff = 0.0;
  workset.n_coeff = 0.0;
  workset.current_time = time;
  /* actually check and abort if problem is unsteady? */
  workset.transientTerms = false;
  workset.accelerationTerms = false;
}

static void setupWorksetNodesetInfo(
    PHAL::Workset& workset,
    RCP<APFDiscretization> d)
{
  workset.nodeSets = rcpFromRef(d->getNodeSets());
  workset.nodeSetCoords = rcpFromRef(d->getNodeSetCoords());
}

void MechAdjResponse::evaluateResponseT(
    const double currentTime,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
  if (evalCtr == 0) {evalCtr++; return; }

  print("solving adjoint problem");

  RCP<Discretization> d =
    rcp(new Discretization(pb->stateManager));

  d->updateSolutionToMesh(xT);

  if (pb->enrich)
    d->enrichDiscretization();

  RCP<LinearSystem> ls = rcp(new LinearSystem(d));

  PHAL::Workset workset;
  setupInitialWorksetInfo(workset,currentTime);
  ls->setWorksetSolutionInfo(workset);
  fmb->evaluateJacobian(workset);

  PHAL::Workset dworkset;
  setupInitialWorksetInfo(dworkset,currentTime);
  setupWorksetNodesetInfo(dworkset,d->getAPFDisc());
  ls->setWorksetDirichletInfo(workset);
  fmb->evaluateDirichletBC(dworkset);

  ls->completeJacobianFill();

  if (writeLinearSystem)
    ls->writeLinearSystem(evalCtr);

  if (pb->enrich)
    d->decreaseDiscretization();

  evalCtr++;
}

}
