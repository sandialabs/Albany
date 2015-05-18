//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_MechAdjResponse.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "PHAL_Utilities.hpp"

namespace GOAL {

typedef AlbanyTraits::Jacobian Jac;
typedef AlbanyTraits::Residual Res;

static void doJacPostReg(
    const RCP<Albany::Application>& app,
    const RCP<Albany::MeshSpecsStruct>& ms,
    RCP<FieldManager<AlbanyTraits> >& jfm)
{
  using PHAL::getDerivativeDimensions;
  std::vector<PHX::index_size_type> dd;
  dd.push_back(getDerivativeDimensions<Jac>(app.get(),ms.get()));
  jfm->setKokkosExtendedDataTypeDimensions<Jac>(dd);
  jfm->postRegistrationSetupForType<Jac>("Jacobian");
}

static void createSingleJFM(
    Teuchos::ParameterList& p,
    const RCP<Albany::Application>& app,
    const RCP<Albany::StateManager>& sm,
    const RCP<Albany::MeshSpecsStruct>& ms,
    RCP<FieldManager<AlbanyTraits> >& jfm)
{
  jfm = rcp(new FieldManager<AlbanyTraits>);
  app->getProblem()->buildEvaluators(
      *jfm,*ms,*sm,Albany::BUILD_RESPONSE_FM,rcp(&p,false));
  doJacPostReg(app,ms,jfm);
}

static void createJFM(
    Teuchos::ParameterList& p,
    const RCP<Albany::Application>& app,
    const RCP<Albany::StateManager>& sm,
    const ArrayRCP<RCP<Albany::MeshSpecsStruct> >& ms,
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& jfm)
{
  int physSets = ms.size();
  jfm.resize(physSets);
  for (int ps=0; ps < physSets; ++ps)
    createSingleJFM(p,app,sm,ms[ps],jfm[ps]);
}

MechAdjResponse::MechAdjResponse(
    const RCP<Albany::Application>& app,
    const RCP<Albany::AbstractProblem>& prob,
    const RCP<Albany::StateManager>& sm,
    const ArrayRCP<RCP<Albany::MeshSpecsStruct> >& ms,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm()),
  app_(app),
  problem_(prob),
  stateMgr_(sm),
  meshSpecs_(ms),
  params_(rp)
{
  out_ = Teuchos::VerboseObjectBase::getDefaultOStream();
  createJFM(params_,app_,stateMgr_,meshSpecs_,jfm_);
}

MechAdjResponse::
~MechAdjResponse()
{
}

void MechAdjResponse::initializeLinearSystem()
{
  map_ = app_->getDiscretization()->getMapT();
  overlapMap_ = app_->getDiscretization()->getOverlapMapT();
  graph_ = app_->getDiscretization()->getJacobianGraphT();
  overlapGraph_ = app_->getDiscretization()->getOverlapJacobianGraphT();
  overlapJac_ = rcp(new Tpetra_CrsMatrix(overlapGraph_));
  overlapJac_->setAllToScalar(0.0);
  exporter_ = rcp(new Tpetra_Export(overlapMap_,map_));
}

void MechAdjResponse::setupWorkset(PHAL::Workset& workset)
{
  initializeLinearSystem();
  workset.JacT = overlapJac_;
  workset.ignore_residual = true;
  /* not adjoint for debugging purposes.
     TODO make this true when debugging is done */
  workset.is_adjoint = false;
  /* TODO don't hard code the below?
     assuming only continuation problems for now */
  workset.j_coeff = 1.0;
  workset.m_coeff = 0.0;
  workset.n_coeff = 0.0;
}

void MechAdjResponse::evaluateJac(PHAL::Workset& workset)
{
  const Albany::WorksetArray<int>::type& wsPhysIndex =
    app_->getDiscretization()->getWsPhysIndex();
  int numWs = app_->getNumWorksets();
  for (int ws=0; ws < numWs; ws++)
  {
    app_->loadWorksetBucketInfo<Jac>(workset,ws);
    jfm_[wsPhysIndex[ws]]->evaluateFields<Jac>(workset);
  }
}

void MechAdjResponse::evaluateResponseT(
    const double currentTime,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
  PHAL::Workset workset;
  app_->setupBasicWorksetInfoT(workset,currentTime,
      rcp(xdotT, false),rcp(xdotdotT, false),rcpFromRef(xT),p);
  workset.gT = Teuchos::rcp(&gT,false);
  setupWorkset(workset);
  evaluateJac(workset);
#if GOAL_DEBUG
  overlapJac_->describe(*out_,Teuchos::VERB_EXTREME);
#endif
}

}
