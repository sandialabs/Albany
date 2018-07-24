//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_TpetraThyraUtils.hpp"

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
		Tpetra_Vector& gT)
{
  Teuchos::ScalarTraits<ST>::magnitudeType twonorm = x->norm_2();
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst(); 
  gT_nonconstView[0] = twonorm;
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
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  if (gT != NULL) {
    Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst(); 
    gT_nonconstView[0] = nrm; 
  }

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  // dg/dx = 1/||x|| * x^T
  Teuchos::ETransp T = Teuchos::TRANS; 
  Teuchos::ETransp N = Teuchos::NO_TRANS; 
  if (gxT != NULL) {
    // Until you change gxT to Thyra, cast x and Vx to Tpetra
    auto xT = Albany::getConstTpetraVector(x);
    if (!Vx.is_null()) {
      auto VxT = Albany::getConstTpetraMultiVector(Vx);
      gxT->multiply(T, N, alpha/nrm, *xT, *VxT, 0.0);
    } else {
      // Until you change gxT to Thyra, cast x to Tpetra
      gxT->update(alpha/nrm, *xT, 0.0);
    }
  }

  if (gpT != NULL) {
    gpT->putScalar(0.0);
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
		Tpetra_Vector* gT,
		Tpetra_MultiVector* dg_dxT,
		Tpetra_MultiVector* dg_dxdotT,
		Tpetra_MultiVector* dg_dxdotdotT,
		Tpetra_MultiVector* dg_dpT)
{
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  Teuchos::ArrayRCP<ST> gT_nonconstView;
  if (gT != NULL) {
    gT_nonconstView = gT->get1dViewNonConst();
    gT_nonconstView[0] = nrm;
  }
  
  // Evaluate dg/dx
  if (dg_dxT != NULL) {
    //double nrm;
    if (gT != NULL) {
      nrm = gT_nonconstView[0];
    } else {
      // Commented this, since it is already compute at the beginning.
      //nrm = x->norm_2();
    }
    // Until you change gxT to Thyra, cast x to Tpetra
    auto xT = Albany::getConstTpetraVector(x);
    dg_dxT->scale(1.0/nrm,*xT);
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
    Tpetra_MultiVector* dg_dpT)
{
  if (dg_dpT != NULL) {
    dg_dpT->putScalar(0.0);
  }
}
