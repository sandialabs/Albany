//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SolutionAverageResponseFunction.hpp"

#include "Albany_TpetraThyraUtils.hpp"

Albany::SolutionAverageResponseFunction::
SolutionAverageResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT) :
  ScalarResponseFunction(commT)
{
}

Albany::SolutionAverageResponseFunction::
~SolutionAverageResponseFunction()
{
}

unsigned int
Albany::SolutionAverageResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::SolutionAverageResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		Tpetra_Vector& gT)
{
  evaluateResponseImpl(*x,gT);
}

void
Albany::SolutionAverageResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
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
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{
  // Evaluate response g
  if (gT != NULL) {
    evaluateResponseImpl(*x,*gT);
  }

  // Evaluate tangent of g = dg/dx*Vx + dg/dxdot*Vxdot + dg/dp*Vp
  // If Vx == NULL, Vx is the identity
  if (gxT != NULL) {
    Teuchos::ArrayRCP<ST> gxT_nonconstView;
    if (!Vx.is_null()) {
      if (ones.is_null() || ones->domain()->dim()!=Vx->domain()->dim()) {
        ones = Thyra::createMembers(Vx->range(), Vx->domain()->dim());
        ones->assign(1.0);
      }
      Teuchos::Array<ST> means; 
      means.resize(Vx->domain()->dim());
      Vx->dots(*ones,means());
      for (auto& mean : means) {
        mean /= Vx->domain()->dim();
      }
      for (int j=0; j<Vx->domain()->dim(); j++) {  
        gxT_nonconstView = gxT->getDataNonConst(j); 
        gxT_nonconstView[0] = means[j];  
      }
    }
    else {
      gxT->putScalar(1.0/x->space()->dim());
    }
    gxT->scale(alpha);
  }
  
  if (gpT != NULL) {
    gpT->putScalar(0.0);
  }
}

void
Albany::SolutionAverageResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* dg_dxT,
		Tpetra_MultiVector* dg_dxdotT,
		Tpetra_MultiVector* dg_dxdotdotT,
		Tpetra_MultiVector* dg_dpT)
{
  // Evaluate response g
  if (gT != NULL) {
    evaluateResponseImpl(*x,*gT);
  }

  // Evaluate dg/dx
  if (dg_dxT != NULL) {
    dg_dxT->putScalar(1.0 / x->space()->dim());
  }

  // Evaluate dg/dxdot
  if (dg_dxdotT != NULL) {
    dg_dxdotT->putScalar(0.0);
  }
  if (dg_dxdotdotT != NULL) {
    dg_dxdotdotT->putScalar(0.0);
  }

  // Evaluate dg/dp
  if (dg_dpT != NULL) {
    dg_dpT->putScalar(0.0);
  }
}

void
Albany::SolutionAverageResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    Tpetra_MultiVector* dg_dpT)
{
  // Evaluate response derivative dg_dp
  if (dg_dpT != NULL) {
    dg_dpT->putScalar(0.0);
  }
}

void 
Albany::SolutionAverageResponseFunction::
evaluateResponseImpl (
    const Thyra_Vector& x,
		Tpetra_Vector& gT)
{
  if (one.is_null()) {
    one = Thyra::createMember(x.space());
    one->assign(1.0);
  }
  const ST mean = one->dot(x) / x.space()->dim();
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst();
  gT_nonconstView[0] = mean; 
}
