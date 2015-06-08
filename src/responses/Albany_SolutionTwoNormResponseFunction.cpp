//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SolutionTwoNormResponseFunction.hpp"

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
evaluateResponseT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 Tpetra_Vector& gT)
{
  Teuchos::ScalarTraits<ST>::magnitudeType twonorm = xT.norm2();
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst(); 
  gT_nonconstView[0] = twonorm;
}

void
Albany::SolutionTwoNormResponseFunction::
evaluateTangentT(const double alpha, 
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
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = xT.norm2();

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
    if (VxT != NULL)
      gxT->multiply(T, N, alpha/nrm, xT, *VxT, 0.0);
    else
      gxT->update(alpha/nrm, xT, 0.0);
  }

  if (gpT != NULL)
    gpT->putScalar(0.0);
}

#if defined(ALBANY_EPETRA)
void
Albany::SolutionTwoNormResponseFunction::
evaluateGradient(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector* xdotdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dxdotdot,
		 Epetra_MultiVector* dg_dp)
{

  // Evaluate response g
  if (g != NULL)
    x.Norm2(&(*g)[0]);

  // Evaluate dg/dx
  if (dg_dx != NULL) {
    double nrm;
    if (g != NULL)
      nrm = (*g)[0];
    else
      x.Norm2(&nrm);
    dg_dx->Scale(1.0/nrm,x);
  }

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);
  if (dg_dxdotdot != NULL)
    dg_dxdotdot->PutScalar(0.0);

  // Evaluate dg/dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}
#endif

void
Albany::SolutionTwoNormResponseFunction::
evaluateGradientT(const double current_time,
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
   Teuchos::ScalarTraits<ST>::magnitudeType nrm = xT.norm2();

  // Evaluate response g
  Teuchos::ArrayRCP<ST> gT_nonconstView;
  if (gT != NULL) {
    gT_nonconstView = gT->get1dViewNonConst();
    gT_nonconstView[0] = nrm;
  }
  
  // Evaluate dg/dx
  if (dg_dxT != NULL) {
    //double nrm;
    if (gT != NULL)
      nrm = gT_nonconstView[0];
    else 
      nrm = xT.norm2();
    dg_dxT->scale(1.0/nrm,xT);
  }

  // Evaluate dg/dxdot
  if (dg_dxdotT != NULL)
    dg_dxdotT->putScalar(0.0);
  if (dg_dxdotdotT != NULL)
    dg_dxdotdotT->putScalar(0.0);

  // Evaluate dg/dp
  if (dg_dpT != NULL)
    dg_dpT->putScalar(0.0);
  
}

#if defined(ALBANY_EPETRA)
//! Evaluate distributed parameter derivative dg/dp
void
Albany::SolutionTwoNormResponseFunction::
evaluateDistParamDeriv(
    const double current_time,
    const Epetra_Vector* xdot,
    const Epetra_Vector* xdotdot,
    const Epetra_Vector& x,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Epetra_MultiVector* dg_dp)
{
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}
#endif // ALBANY_EPETRA
