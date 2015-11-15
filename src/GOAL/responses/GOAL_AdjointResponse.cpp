//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponse.hpp"
#include "GOAL_MechanicsProblem.hpp"
#include "Albany_GOALDiscretization.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "PHAL_Utilities.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace GOAL {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::ParameterList;

static void print(const char* msg)
{
  RCP<Teuchos::FancyOStream> out =
    Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << msg << std::endl;
}

AdjointResponse::AdjointResponse(
    RCP<Albany::Application> const& app,
    RCP<Albany::AbstractProblem> const& prob,
    RCP<Albany::StateManager> const& sm,
    ArrayRCP<RCP<Albany::MeshSpecsStruct> > const& ms,
    ParameterList& rp) :
  ScalarResponseFunction(app->getComm()),
  evalCtr(0),
  time(0),
  enrichAdjoint(false),
  application(app),
  stateManager(sm),
  meshSpecs(ms),
  params(rp)
{
  print("Building adjoint pde instantiations");
  problem = Teuchos::rcp_dynamic_cast<Albany::GOALMechanicsProblem>(prob);
  enrichAdjoint = problem->enrichAdjoint;
  buildFieldManagers();
}

AdjointResponse::~AdjointResponse()
{
}

void AdjointResponse::buildFieldManagers()
{
  problem->isAdjoint = true;
  int physSets = meshSpecs.size();
  fm.resize(physSets);
  for (int ps=0; ps < physSets; ++ps)
  {
    fm[ps] = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    problem->buildEvaluators(*fm[ps], *meshSpecs[ps], *stateManager,
        Albany::BUILD_RESPONSE_FM, rcp(&params, false));
  }
  problem->isAdjoint = false;
}

static RCP<Albany::GOALDiscretization> getDiscretization(
    RCP<Albany::Application> const& app)
{
  RCP<Albany::AbstractDiscretization> d = app->getDiscretization();
  RCP<Albany::GOALDiscretization> gd =
    Teuchos::rcp_dynamic_cast<Albany::GOALDiscretization>(d);
  return gd;
}

void AdjointResponse::postRegistrationSetup()
{
  typedef PHAL::AlbanyTraits::Jacobian J;
  for (int ps=0; ps < meshSpecs.size(); ++ps)
  {
    std::vector<PHX::index_size_type> dd;
    dd.push_back(PHAL::getDerivativeDimensions<J>(application.get(), ps));
    fm[ps]->setKokkosExtendedDataTypeDimensions<J>(dd);
    fm[ps]->postRegistrationSetupForType<J>("Jacobian");
  }
}

void AdjointResponse::initializeLinearSystem()
{
  RCP<const Tpetra_Map> m = discretization->getMapT();
  RCP<const Tpetra_Map> om = discretization->getOverlapMapT();
  RCP<const Tpetra_CrsGraph> g = discretization->getJacobianGraphT();
  RCP<const Tpetra_CrsGraph> og = discretization->getOverlapJacobianGraphT();
  x = rcp(new Tpetra_Vector(m));
  overlapX = rcp(new Tpetra_Vector(om));
  qoi = rcp(new Tpetra_Vector(m));
  overlapQoI = rcp(new Tpetra_Vector(om));
  jac = rcp(new Tpetra_CrsMatrix(g));
  overlapJac = rcp(new Tpetra_CrsMatrix(og));
  importer = rcp(new Tpetra_Import(m, om));
  exporter = rcp(new Tpetra_Export(om, m));
  dummy = rcp(new Tpetra_Vector(om));
  discretization->fillSolutionVector(overlapX);
}

void AdjointResponse::initializeWorkset(PHAL::Workset& workset)
{
  workset.is_adjoint = false;
  workset.j_coeff = 1.0;
  workset.m_coeff = 0.0;
  workset.n_coeff = 0.0;
  workset.current_time = time;
  workset.transientTerms = false;
  workset.accelerationTerms = false;
  workset.xT = overlapX;
  workset.xdotT = dummy;
  workset.xdotdotT = dummy;
  workset.qoi = overlapQoI;
  workset.JacT = overlapJac;
  workset.x_importerT = importer;
  workset.comm = x->getMap()->getComm();
}

void AdjointResponse::fillLinearSystem()
{
  typedef PHAL::AlbanyTraits::Jacobian J;
  PHAL::Workset workset;
  initializeWorkset(workset);
  const Albany::WorksetArray<int>::type& wsPhysIndex =
    discretization->getWsPhysIndex();
  int numWs = application->getNumWorksets();
  for (int ws=0; ws < numWs; ++ws)
  {
    application->loadWorksetBucketInfo<J>(workset, ws);
    fm[wsPhysIndex[ws]]->evaluateFields<J>(workset);
  }
  jac->doExport(*overlapJac, *exporter, Tpetra::ADD);
  qoi->doExport(*overlapQoI, *exporter, Tpetra::ADD);
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
  print("Solving adjoint problem");
  time = currentTime;
  discretization = getDiscretization(application);
  discretization->attachSolutionToMesh(xT);
  postRegistrationSetup();
  initializeLinearSystem();
  fillLinearSystem();
  evalCtr++;
}

}
