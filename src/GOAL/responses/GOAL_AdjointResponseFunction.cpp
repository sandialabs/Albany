//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_AdjointResponseFunction.hpp"
#include "PHAL_Utilities.hpp"

namespace GOAL {

AdjointResponseFunction::AdjointResponseFunction(
    const Teuchos::RCP<Albany::Application>& application,
    const Teuchos::RCP<Albany::AbstractProblem>& problem,
    const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs,
    const Teuchos::RCP<Albany::StateManager>& stateManager,
    Teuchos::ParameterList& responseParams) :
  ScalarResponseFunction(application->getComm()),
  application_(application),
  problem_(problem),
  meshSpecs_(meshSpecs),
  stateManager_(stateManager),
  responseParams_(responseParams)
{
  setupT();
}

AdjointResponseFunction::~AdjointResponseFunction()
{
}

void AdjointResponseFunction::setupT()
{
  // create field manager
  rfm_ = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  // create evaluators for field manager
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > tags =
    problem_->buildEvaluators(
        *rfm_,
        *meshSpecs_,
        *stateManager_,
        Albany::BUILD_RESPONSE_FM,
        Teuchos::rcp(&responseParams_,false));

  // visualize response field manager graph
  visResponseGraph_ =
    responseParams_.get("Phalanx Graph Visualization Detail", 0);
  visResponseName_ = responseParams_.get<std::string>("Name");
  std::replace(visResponseName_.begin(), visResponseName_.end(), ' ', '_');
  std::transform(
      visResponseName_.begin(),
      visResponseName_.end(),
      visResponseName_.begin(),
      ::tolower);
}

Teuchos::RCP<const Tpetra_Map> AdjointResponseFunction::responseMapT() const
{
}

bool AdjointResponseFunction::isScalarResponse() const
{
  return true;
}

unsigned int AdjointResponseFunction::numResponses() const
{
  return 1;
}

Teuchos::RCP<Tpetra_Operator> AdjointResponseFunction::createGradientOpT() const
{
}

void AdjointResponseFunction::evaluateResponseT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
}

void AdjointResponseFunction::evaluateTangentT(
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

void AdjointResponseFunction::evaluateDerivativeT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    Tpetra_Vector* gT,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{
  if (gT) this->evaluateResponseT(current_time, xdotT, xdotdotT, xT, p, *gT);
}

void AdjointResponseFunction::evaluateGradientT(
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
