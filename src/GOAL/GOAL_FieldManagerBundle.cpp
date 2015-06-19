//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_BCManager.hpp"
#include "GOAL_FieldManagerBundle.hpp"
#include "Albany_Application.hpp"
#include "PHAL_Utilities.hpp"

namespace GOAL {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::rcpFromRef;
using Albany::Application;
using Albany::MeshSpecsStruct;
using PHX::FieldManager;
using PHAL::AlbanyTraits;

typedef AlbanyTraits::Jacobian J;

static void enrichMeshSpecs(
    ArrayRCP<RCP<MeshSpecsStruct> >& oldMs)
{
  ArrayRCP<RCP<MeshSpecsStruct> > newMs;
  int physSet = oldMs.size();
  newMs.resize(physSet);
  for (int ps=0; ps < physSet; ++ps)
  {
    const char* name = oldMs[ps]->ctd.name;
    assert (strcmp(name,"Tetrahedron_4")==0);
    const CellTopologyData* ctd =
      shards::getCellTopologyData<shards::Tetrahedron<10> >();
    newMs[ps] = rcp(new MeshSpecsStruct(
          *ctd,
          oldMs[ps]->numDim,
          oldMs[ps]->cubatureDegree,
          oldMs[ps]->nsNames,
          oldMs[ps]->ssNames,
          oldMs[ps]->worksetSize,
          oldMs[ps]->ebName,
          oldMs[ps]->ebNameToIndex,
          oldMs[ps]->interleavedOrdering,
          oldMs[ps]->sepEvalsByEB,
          oldMs[ps]->cubatureRule));
  }
  oldMs = newMs;
}

FieldManagerBundle::FieldManagerBundle(
    RCP<BCManager>& mgr,
    RCP<ProblemBundle>& bundle) :
  pb(bundle),
  bcm(mgr)
{
  if (pb->enrich)
    enrichMeshSpecs(pb->meshSpecs);
  createFieldManagers();
}

FieldManagerBundle::~FieldManagerBundle()
{
}

template<typename EvalT>
static void doPostReg(
    std::string eval,
    RCP<FieldManager<AlbanyTraits> >& fm)
{
  fm->postRegistrationSetupForType<EvalT>(eval);
}

static void setJacobianDerivDims(
    RCP<Application>& app,
    RCP<MeshSpecsStruct>& ms,
    RCP<FieldManager<AlbanyTraits> >& fm)
{
  using PHAL::getDerivativeDimensions;
  std::vector<PHX::index_size_type> dd;
  dd.push_back(getDerivativeDimensions<J>(app.get(),ms.get()));
  fm->setKokkosExtendedDataTypeDimensions<J>(dd);
}

static void createProblemFieldManager(
    RCP<ProblemBundle>& pb,
    RCP<MeshSpecsStruct>& ms,
    RCP<FieldManager<AlbanyTraits> >& fm)
{
  fm = rcp( new FieldManager<AlbanyTraits> );
  pb->problem->buildEvaluators(
      *fm,*ms,*(pb->stateManager),
      Albany::BUILD_RESPONSE_FM,
      rcp(&(pb->params),false));
  setJacobianDerivDims(pb->application,ms,fm);
  doPostReg<J>("Jacobian",fm);
}

static void createProblemFieldManagers(
    RCP<ProblemBundle>& pb,
    ArrayRCP<RCP<FieldManager<AlbanyTraits> > >& fm)
{
  int physSets = pb->meshSpecs.size();
  fm.resize(physSets);
  for (int ps=0; ps < physSets; ++ps)
    createProblemFieldManager(pb,pb->meshSpecs[ps],fm[ps]);
}

static void createDirichletFieldManager(
    RCP<BCManager>& bcm,
    RCP<ProblemBundle>& pb,
    RCP<FieldManager<AlbanyTraits> >& dfm)
{
  Albany::BCUtils<Albany::DirichletTraits> du;
  dfm = du.constructBCEvaluators(
      pb->meshSpecs[0]->nsNames,
      bcm->dirichletNames,
      rcpFromRef(bcm->params),
      bcm->paramLib);
}

void FieldManagerBundle::createFieldManagers()
{
  createProblemFieldManagers(pb,fm);
  createDirichletFieldManager(bcm,pb,dfm);
}

template<typename EvalT>
static void writePhalanxGraph(
    int ps,
    const char* eval,
    RCP<FieldManager<AlbanyTraits> >& fm)
{
  char name[20];
  sprintf(name,"g_%s_%i",eval,ps);
  fm->writeGraphvizFile<EvalT>(name,2,2);
}

void FieldManagerBundle::writePHXGraphs()
{
  int physSet = fm.size();
  for (int ps=0; ps < physSet; ++ps)
    writePhalanxGraph<J>(ps,"jac",fm[ps]);
}

void FieldManagerBundle::evaluateJacobian(PHAL::Workset& workset)
{
  const Albany::WorksetArray<int>::type& wsPhysIndex =
    pb->application->getDiscretization()->getWsPhysIndex();
  int numWs = pb->application->getNumWorksets();
  for (int ws=0; ws < numWs; ws++)
  {
    pb->application->loadWorksetBucketInfo<J>(workset,ws);
    fm[wsPhysIndex[ws]]->evaluateFields<J>(workset);
  }
}

}
