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
    Tpetra_Vector* gT,
    Tpetra_MultiVector* /*gxT*/,
    Tpetra_MultiVector* /*gpT*/)
{
  // Evaluate just the response if it is requested.
  if (gT) {
    this->evaluateResponse(current_time, x, xdot, xdotdot, p, *gT);
  }
}

void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* /*deriv_p*/,
    Tpetra_Vector* gT,
    Tpetra_MultiVector* /*dg_dxT*/,
    Tpetra_MultiVector* /*dg_dxdotT*/,
    Tpetra_MultiVector* /*dg_dxdotdotT*/,
    Tpetra_MultiVector* /*dg_dpT*/)
{
  if (gT) {
    this->evaluateResponse(current_time, x, xdot, xdotdot, p, *gT);
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
    Tpetra_MultiVector* /*dg_dpT*/)
{
  // Do nothing
}
