//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SolutionAverageResponseFunction.hpp"

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
evaluateResponseT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 Tpetra_Vector& gT)
{
  ST mean = xT.meanValue();
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst();
  gT_nonconstView[0] = mean; 
}


void
Albany::SolutionAverageResponseFunction::
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
  // Evaluate response g
  if (gT != NULL) {
    ST mean = xT.meanValue();
    Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst();
    gT_nonconstView[0] = mean; 
  }

  // Evaluate tangent of g = dg/dx*Vx + dg/dxdot*Vxdot + dg/dp*Vp
  // If Vx == NULL, Vx is the identity
  if (gxT != NULL) {
    Teuchos::ArrayRCP<ST> gxT_nonconstView;
    if (VxT != NULL) {
       Teuchos::Array<ST> means; 
       means.resize(VxT->getNumVectors());
       VxT->meanValue(means());  
      for (int j=0; j<VxT->getNumVectors(); j++) {  
        gxT_nonconstView = gxT->getDataNonConst(j); 
        gxT_nonconstView[0] = means[j];  
      }
    }
    else {
      gxT->putScalar(1.0/xT.getGlobalLength());
    }
    gxT->scale(alpha);
  }
  
  if (gpT != NULL)
    gpT->putScalar(0.0);
}

#if defined(ALBANY_EPETRA)
void
Albany::SolutionAverageResponseFunction::
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
    x.MeanValue(&(*g)[0]);

  // Evaluate dg/dx
  if (dg_dx != NULL)
    dg_dx->PutScalar(1.0 / x.GlobalLength());

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
Albany::SolutionAverageResponseFunction::
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

  // Evaluate response g
  if (gT != NULL) {
    ST mean = xT.meanValue();
    Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst();
    gT_nonconstView[0] = mean;
  }

  // Evaluate dg/dx
  if (dg_dxT != NULL)
    dg_dxT->putScalar(1.0 / xT.getGlobalLength());

  // Evaluate dg/dxdot
  if (dg_dxdotT != NULL)
    dg_dxdotT->putScalar(0.0);
  if (dg_dxdotdotT != NULL)
    dg_dxdotdotT->putScalar(0.0);

  // Evaluate dg/dp
  if (dg_dpT != NULL)
    dg_dpT->putScalar(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateDistParamDerivT(
         const double current_time,
         const Tpetra_Vector* xdotT,
         const Tpetra_Vector* xdotdotT,
         const Tpetra_Vector& xT,
         const Teuchos::Array<ParamVec>& param_array,
         const std::string& dist_param_name,
         Tpetra_MultiVector* dg_dpT) {
  // Evaluate response derivative dg_dp
  if (dg_dpT != NULL)
    dg_dpT->putScalar(0.0);
}
