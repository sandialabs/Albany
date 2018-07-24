//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_KLResponseFunction.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_VerboseObject.hpp"

Albany::KLResponseFunction::
KLResponseFunction(
  const Teuchos::RCP<Albany::AbstractResponseFunction>& response_,
  Teuchos::ParameterList& responseParams) :
  response(response_),
  responseParams(responseParams),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  num_kl = responseParams.get("Number of KL Terms", 5);
}

Albany::KLResponseFunction::
~KLResponseFunction()
{
}

Teuchos::RCP<const Tpetra_Map>
Albany::KLResponseFunction::
responseMapT() const
{
  return response->responseMapT();
}

Teuchos::RCP<Tpetra_Operator>
Albany::KLResponseFunction::
createGradientOpT() const
{
  return response->createGradientOpT();
}

bool
Albany::KLResponseFunction::
isScalarResponse() const
{
  return response->isScalarResponse();
}


void
Albany::KLResponseFunction::
evaluateResponse(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    Tpetra_Vector& gT)
{
  response->evaluateResponse(current_time, x, xdot, xdotdot, p, gT);
}


void
Albany::KLResponseFunction::
evaluateTangent(const double alpha,
    const double beta,
    const double omega,
    const double current_time,
    bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
    Tpetra_Vector* gT,
    Tpetra_MultiVector* gxT,
    Tpetra_MultiVector* gpT)
{
  response->evaluateTangent(alpha, beta, omega, current_time, sum_derivs,
          x, xdot, xdotdot, p, deriv_p, Vx, Vxdot, Vxdotdot, Vp,
          gT, gxT, gpT);
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::KLResponseFunction::
evaluateDistParamDeriv(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Tpetra_MultiVector*  dg_dpT)
{
  response->evaluateDistParamDeriv(current_time, x, xdot, xdotdot, param_array, dist_param_name, dg_dpT);
}

void
Albany::KLResponseFunction::
evaluateDerivative(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    Tpetra_Vector* gT,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxT,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotT,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdotT,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dpT)
{
  response->evaluateDerivative(current_time, x, xdot, xdotdot, p, deriv_p,
             gT, dg_dxT, dg_dxdotT, dg_dxdotdotT, dg_dpT);
}
