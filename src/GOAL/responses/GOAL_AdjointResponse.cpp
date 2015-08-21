//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponse.hpp"
#include "GOAL_Discretization.hpp"
#include "GOAL_MechanicsProblem.hpp"
#include "Albany_APFDiscretization.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Utilities.hpp"
#include <PCU.h>

namespace GOAL {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::rcpFromRef;
using Teuchos::rcp_dynamic_cast;

using Albany::Application;
using Albany::AbstractProblem;
using Albany::StateManager;
using Albany::MeshSpecsStruct;

using Albany::APFDiscretization;

using PHX::FieldManager;
using PHAL::AlbanyTraits;

typedef AlbanyTraits::Residual R;
typedef AlbanyTraits::Jacobian J;

/*****************************************************************************/
static void print(const char* msg, ...)
{
  if (PCU_Comm_Self())
    return void();
  printf("GOAL: ");
  va_list ap;
  va_start(ap, msg);
  vfprintf(stdout, msg, ap);
  va_end(ap);
  printf("\n");
}

/*****************************************************************************/
AdjointResponse::AdjointResponse(
    const RCP<Application>& app,
    const RCP<AbstractProblem>& prob,
    const RCP<StateManager>& sm,
    const ArrayRCP<RCP<MeshSpecsStruct> >& ms,
    Teuchos::ParameterList& rp) :
  evalCtr(0),
  application(app),
  ScalarResponseFunction(app->getComm())
{
  RCP<Albany::GOALMechanicsProblem> problem = 
    rcp_dynamic_cast<Albany::GOALMechanicsProblem>(prob);

  problem->buildAdjointProblem(ms, *sm, rcp(&rp, false));

  fm = problem->getAdjointFieldManager();
  dfm = problem->getAdjointDirichletFieldManager();
  qfm = problem->getAdjointQoIFieldManager();
}

/*****************************************************************************/
AdjointResponse::~AdjointResponse()
{
}

/*****************************************************************************/
static void jacobianPostRegistration(
    RCP<Application>& app,
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& fm)
{
  for (int ps=0; ps < fm.size(); ++ps)
  {
    std::vector<PHX::index_size_type> dd;
    dd.push_back(PHAL::getDerivativeDimensions<J>(app.get(), ps));
    fm[ps]->setKokkosExtendedDataTypeDimensions<J>(dd);
    fm[ps]->postRegistrationSetupForType<J>("Jacobian");
  }
}

static void qoiPostRegistration(
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& qfm)
{
}

static void dirichletPostRegistration(
    RCP<Application>& app,
    RCP<FieldManager<AlbanyTraits> >& dfm)
{
}

static void postRegistration(
    RCP<Application>& app,
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& fm,
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& qfm)
{
  jacobianPostRegistration(app, fm);
  qoiPostRegistration(qfm);
}

/*****************************************************************************/
struct LinearSystem
{
  RCP<Tpetra_Vector> x;
  RCP<Tpetra_Vector> qoi;
  RCP<Tpetra_CrsMatrix> jac;
  RCP<Tpetra_Vector> overlapX;
  RCP<Tpetra_Vector> overlapQoI;
  RCP<Tpetra_CrsMatrix> overlapJac;
  RCP<Tpetra_Import> importer;
  RCP<Tpetra_Export> exporter;
  RCP<Tpetra_Vector> dummy;
};

static void initializeLinearSystem(
    RCP<LinearSystem>& ls,
    RCP<APFDiscretization>& d)
{
  RCP<const Tpetra_Map> m = d->getMapT();
  RCP<const Tpetra_Map> om = d->getOverlapMapT();
  RCP<const Tpetra_CrsGraph> g = d->getJacobianGraphT();
  RCP<const Tpetra_CrsGraph> og = d->getOverlapJacobianGraphT();
  ls->x = rcp(new Tpetra_Vector(m));
  ls->qoi = rcp(new Tpetra_Vector(m));
  ls->jac = rcp(new Tpetra_CrsMatrix(g));
  ls->overlapX = rcp(new Tpetra_Vector(om));
  ls->overlapQoI = rcp(new Tpetra_Vector(om));
  ls->overlapJac = rcp(new Tpetra_CrsMatrix(og));
  ls->importer = rcp(new Tpetra_Import(m, om));
  ls->exporter = rcp(new Tpetra_Export(om, m));
  ls->dummy = rcp(new Tpetra_Vector(om));
}

RCP<LinearSystem> createLinearSystem(
    RCP<Discretization>& d)
{
  RCP<LinearSystem> ls = rcp(new LinearSystem);
  initializeLinearSystem(ls, d->disc);
  d->getSolution(ls->x);
  ls->overlapX->doImport(*(ls->x), *(ls->importer), Tpetra::INSERT);
  return ls;
}

/*****************************************************************************/
static void setWorksetSolutionInfo(
    PHAL::Workset& workset,
    RCP<LinearSystem>& ls)
{
  workset.xT = ls->overlapX;
  workset.xdotT = ls->dummy;
  workset.xdotdotT = ls->dummy;
  workset.JacT = ls->overlapJac;
  workset.x_importerT = ls->importer;
  workset.comm = ls->x->getMap()->getComm();
}

static void setWorksetJacobianInfo(
    PHAL::Workset& workset,
    const double time)
{
  workset.is_adjoint = false;
  workset.j_coeff = 1.0;
  workset.m_coeff = 0.0;
  workset.n_coeff = 0.0;
  workset.current_time = time;
  workset.transientTerms = false;
  workset.accelerationTerms = false;
  workset.ignore_residual = true;
}

/*****************************************************************************/
static void evaluateJacobian(
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& fm,
    RCP<Application>& app,
    RCP<LinearSystem>& ls,
    const double time)
{
  double t0 = PCU_Time();
  PHAL::Workset workset;
  setWorksetJacobianInfo(workset, time);
  setWorksetSolutionInfo(workset, ls);
  const Albany::WorksetArray<int>::type& wsPhysIndex =
    app->getDiscretization()->getWsPhysIndex();
  int numWs = app->getNumWorksets();
  for (int ws=0; ws < numWs; ++ws)
  {
    app->loadWorksetBucketInfo<J>(workset, ws);
    fm[wsPhysIndex[ws]]->evaluateFields<J>(workset);
  }
  double t1 = PCU_Time();
  print("jacobian assembled in %f seconds", t1-t0);
}

static void evaluateQoI(
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& qfm,
    RCP<Application>& app,
    RCP<LinearSystem>& ls,
    const double time)
{
}

/*****************************************************************************/
void AdjointResponse::evaluateResponseT(
    const double currentTime,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
  // skip the (usually) trivial 0th continuation step
  if (evalCtr == 0) {evalCtr++; return;}
  print("solving adjoint problem");

  // perform post registration setup for field managers
  postRegistration(application, fm, qfm);

  // create a discretization with the updated solution
  RCP<StateManager> sm = rcpFromRef(application->getStateMgr());
  RCP<Discretization> d = rcp(new Discretization(sm));
  d->updateSolutionToMesh(xT);

  // create a linear system and fill the solution vector
  RCP<LinearSystem> ls = createLinearSystem(d);

  // evaluate the field managers
  evaluateJacobian(fm, application, ls, currentTime);

  evalCtr++;
}

}
