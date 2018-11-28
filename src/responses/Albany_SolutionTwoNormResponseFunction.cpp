//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionTwoNormResponseFunction.hpp"

#include "Thyra_VectorStdOps.hpp"

Albany::SolutionTwoNormResponseFunction::
SolutionTwoNormResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT) :
  SamplingBasedScalarResponseFunction(commT)
{
}

Albany::SolutionTwoNormResponseFunction::
~SolutionTwoNormResponseFunction()
{
}

unsigned int
Albany::SolutionTwoNormResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::SolutionTwoNormResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const Teuchos::RCP<Thyra_Vector>& g)
{
  Teuchos::ScalarTraits<ST>::magnitudeType twonorm = x->norm_2();
  g->assign(twonorm);
}

void
Albany::SolutionTwoNormResponseFunction::
evaluateTangent(const double alpha, 
		const double /*beta*/,
		const double /*omega*/,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  if (!g.is_null()) {
    g->assign(nrm);
  }

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  // dg/dx = 1/||x|| * x^T
  if (!gx.is_null()) {
    if (!Vx.is_null()) {
      // compute gx = x' * Vx. x is a vector, Vx a multivector,
      // so gx is a MV with range->dim()=1, each column being
      // the dot product x->dot(*Vx->col(j))
      for (int j=0; j<Vx->domain()->dim(); ++j) {
        gx->col(j)->assign(x->dot(*Vx->col(j)));
      }
    } else {
      // V_StV stands for V_out = Scalar * V_in
      Thyra::V_StV(gx->col(0).ptr(),alpha/nrm,*x);
    }
  }

  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

void
Albany::SolutionTwoNormResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
		const Teuchos::RCP<Thyra_Vector>& g,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  if (!g.is_null()) {
    g->assign(nrm);
  }
  
  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    // V_StV stands for V_out = Scalar * V_in
    Thyra::V_StV(dg_dx->col(0).ptr(),1.0/nrm,*x);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::SolutionTwoNormResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}
