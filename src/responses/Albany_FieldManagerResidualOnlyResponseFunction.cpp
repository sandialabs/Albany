//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_FieldManagerResidualOnlyResponseFunction.hpp"

Albany::FieldManagerResidualOnlyResponseFunction::
FieldManagerResidualOnlyResponseFunction (
  const Teuchos::RCP<Albany::Application>& application_,
  const Teuchos::RCP<Albany::AbstractProblem>& problem_,
  const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<Albany::StateManager>& stateMgr_,
  Teuchos::ParameterList& responseParams) :
  FieldManagerScalarResponseFunction(application_, problem_, meshSpecs_, stateMgr_,
                                     responseParams)
{}

void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateTangent(const double /*alpha*/, 
    const double /*beta*/,
    const double /*omega*/,
    const double current_time,
    bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* /*deriv_p*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vx*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& /*gx*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*gp*/)
{
  // Evaluate just the response if it is requested.
  if (!g.is_null()) {
    this->evaluateResponse(current_time, x, xdot, xdotdot, p, g);
  }
}

void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* /*deriv_p*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dx*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dxdot*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dxdotdot*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dp*/)
{
  if (!g.is_null()) {
    this->evaluateResponse(current_time, x, xdot, xdotdot, p, g);
  }
}

void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& /*dg_dp*/)
{
  // Do nothing
}
