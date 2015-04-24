//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "GOAL_AdjointResponseFunction.hpp"
#if defined(ALBANY_EPETRA)
#include "Petra_Converters.hpp"
#endif
#include <algorithm>
#include "PHAL_Utilities.hpp"

namespace GOAL {

AdjointResponseFunction::
AdjointResponseFunction(
    const Teuchos::RCP<Albany::Application>& app,
    const Teuchos::RCP<Albany::AbstractProblem>& prob,
    const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
    const Teuchos::RCP<Albany::StateManager>& sm,
    Teuchos::ParameterList& rp) :
  ScalarResponseFunction(app->getComm()),
  application_(app),
  problem_(prob),
  meshSpecs_(ms),
  stateManager_(sm)
{
    setup(rp);
}

AdjointResponseFunction::
AdjointResponseFunction(
    const Teuchos::RCP<Albany::Application>& app,
    const Teuchos::RCP<Albany::AbstractProblem>& prob,
    const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
    const Teuchos::RCP<Albany::StateManager>& sm) :
  ScalarResponseFunction(app->getComm()),
  application_(app),
  problem_(prob),
  meshSpecs_(ms),
  stateManager_(sm)
{
}

void AdjointResponseFunction::
setup(Teuchos::ParameterList& responseParams)
{
  Teuchos::RCP<const Teuchos_Comm> commT = application_->getComm();

  // Restrict to the element block?
  const char* reb_parm = "Restrict to Element Block";
  const bool
    reb_parm_present = responseParams.isType<bool>(reb_parm),
    reb = reb_parm_present && responseParams.get<bool>(reb_parm, false);
  elemBlockIdx_ = reb ? meshSpecs_->ebNameToIndex[meshSpecs_->ebName] : -1;
  if (reb_parm_present) responseParams.remove(reb_parm, false);

  // Create field manager
  rfm_ = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  // Create evaluators for field manager
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > tags =
    problem_->buildEvaluators(
        *rfm_,
        *meshSpecs_,
        *stateManager_,
        Albany::BUILD_RESPONSE_FM,
        Teuchos::rcp(&responseParams,false));

  int rank = tags[0]->dataLayout().rank();
  numResponses_ = tags[0]->dataLayout().dimension(rank-1);
  if (numResponses_ == 0)
    numResponses_ = 1;

  // Do post-registration setup

  // this is not right because rfm doesn't account for multiple element
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
        application_.get(), meshSpecs_.get()));
    rfm_->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
      derivative_dimensions); }
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
        application_.get(), meshSpecs_.get()));
    rfm_->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(
      derivative_dimensions); }
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
        application_.get(), meshSpecs_.get()));
    rfm_->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
      derivative_dimensions); }
  rfm_->postRegistrationSetup("");

  if (reb_parm_present) responseParams.set<bool>(reb_parm, reb);
}

AdjointResponseFunction::
~AdjointResponseFunction()
{
}

unsigned int AdjointResponseFunction::
numResponses() const
{
  return numResponses_;
}

template<typename EvalT>
void AdjointResponseFunction::
evaluate(PHAL::Workset& workset)
{
  const Albany::WorksetArray<int>::type&
    wsPhysIndex = application_->getDiscretization()->getWsPhysIndex();
  rfm_->preEvaluate<EvalT>(workset);
  for (int ws = 0, numWorksets = application_->getNumWorksets();
       ws < numWorksets; ws++) {
    if (elemBlockIdx_ >= 0 && elemBlockIdx_ != wsPhysIndex[ws])
      continue;
    application_->loadWorksetBucketInfo<EvalT>(workset, ws);
    rfm_->evaluateFields<EvalT>(workset);
  }
  rfm_->postEvaluate<EvalT>(workset);
}

void AdjointResponseFunction::
evaluateResponseT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
  // Set data in Workset struct
  PHAL::Workset workset;

  application_->setupBasicWorksetInfoT(workset, current_time, rcp(xdotT, false), rcp(xdotdotT, false), rcpFromRef(xT), p);
  workset.gT = Teuchos::rcp(&gT,false);

  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Residual>(workset);
}


void AdjointResponseFunction::
evaluateTangentT(
    const double alpha, 
    const double beta,
    const double omega,
    const double current_time,
    bool sum_derivs,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Tpetra_MultiVector* VxdotT,
    const Tpetra_MultiVector* VxdotdotT,
    const Tpetra_MultiVector* VxT,
    const Tpetra_MultiVector* VpT,
    Tpetra_Vector* gT,
    Tpetra_MultiVector* gxT,
    Tpetra_MultiVector* gpT)
{
  if (gT) this->evaluateResponseT(current_time, xdotT, xdotdotT, xT, p, *gT);
}

void AdjointResponseFunction::
evaluateGradientT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    Tpetra_Vector* gT,
    Tpetra_MultiVector* dg_dxT,
    Tpetra_MultiVector* dg_dxdotT,
    Tpetra_MultiVector* dg_dxdotdotT,
    Tpetra_MultiVector* dg_dpT)
{
  if (gT) this->evaluateResponseT(current_time, xdotT, xdotdotT, xT, p, *gT);
}

}
