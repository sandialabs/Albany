//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

#if defined(ALBANY_EPETRA)
void
Albany::SolutionAverageResponseFunction::
evaluateDistParamDeriv(
         const double current_time,
         const Epetra_Vector* xdot,
         const Epetra_Vector* xdotdot,
         const Epetra_Vector& x,
         const Teuchos::Array<ParamVec>& param_array,
         const std::string& dist_param_name,
         Epetra_MultiVector* dg_dp) {
  // Evaluate response derivative dg_dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}
#endif

#ifdef ALBANY_SG
void
Albany::SolutionAverageResponseFunction::
evaluateSGResponse(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g)
{
  for (int i=0; i<sg_x.size(); i++)
    sg_x[i].MeanValue(&sg_g[i][0]);
}

void
Albany::SolutionAverageResponseFunction::
evaluateSGTangent(
  const double alpha, 
  const double beta, 
  const double omega, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp)
{
  // Evaluate response g
  if (sg_g != NULL)
    for (int i=0; i<sg_x.size(); i++)
      sg_x[i].MeanValue(&(*sg_g)[i][0]);

  // Evaluate tangent of g = dg/dx*Vx + dg/dxdot*Vxdot + dg/dp*Vp
  // If Vx == NULL, Vx is the identity
  if (sg_JV != NULL) {
    sg_JV->init(0.0);
    if (Vx != NULL)
      for (int j=0; j<Vx->NumVectors(); j++)
	(*Vx)(j)->MeanValue(&(*sg_JV)[0][j][0]);
    else
      (*sg_JV)[0].PutScalar(alpha/sg_x[0].GlobalLength());
  }
  if (sg_gp != NULL)
    sg_gp->init(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateSGGradient(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp)
{
  // Evaluate response g
  if (sg_g != NULL)
    for (int i=0; i<sg_x.size(); i++)
      sg_x[i].MeanValue(&(*sg_g)[i][0]);

  // Evaluate dg/dx
  if (sg_dg_dx != NULL)
    (*sg_dg_dx)[0].PutScalar(1.0 / sg_x[0].GlobalLength());

  // Evaluate dg/dxdot
  if (sg_dg_dxdot != NULL)
    sg_dg_dxdot->init(0.0);
  if (sg_dg_dxdotdot != NULL)
    sg_dg_dxdotdot->init(0.0);

  // Evaluate dg/dp
  if (sg_dg_dp != NULL)
    sg_dg_dp->init(0.0);
}
#endif 
#ifdef ALBANY_ENSEMBLE 

void
Albany::SolutionAverageResponseFunction::
evaluateMPResponse(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g)
{
  for (int i=0; i<mp_x.size(); i++)
    mp_x[i].MeanValue(&mp_g[i][0]);
}

void
Albany::SolutionAverageResponseFunction::
evaluateMPTangent(
  const double alpha, 
  const double beta, 
  const double omega, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_JV,
  Stokhos::ProductEpetraMultiVector* mp_gp)
{
  // Evaluate response g
  if (mp_g != NULL)
    for (int i=0; i<mp_x.size(); i++)
      mp_x[i].MeanValue(&(*mp_g)[i][0]);

  // Evaluate tangent of g = dg/dx*Vx + dg/dxdot*Vxdot + dg/dp*Vp
  // If Vx == NULL, Vx is the identity
  if (mp_JV != NULL) {
    if (Vx != NULL)
      for (int i=0; i<mp_x.size(); i++)
	for (int j=0; j<Vx->NumVectors(); j++)
	  (*Vx)(j)->MeanValue(&(*mp_JV)[i][j][0]);
    else
      for (int i=0; i<mp_x.size(); i++)
	(*mp_JV)[i].PutScalar(alpha/mp_x[0].GlobalLength());
  }
  if (mp_gp != NULL)
    mp_gp->init(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateMPGradient(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_dg_dx,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dp)
{
  // Evaluate response g
  if (mp_g != NULL)
    for (int i=0; i<mp_x.size(); i++)
      mp_x[i].MeanValue(&(*mp_g)[i][0]);

  // Evaluate dg/dx
  if (mp_dg_dx != NULL)
    for (int i=0; i<mp_x.size(); i++)
      (*mp_dg_dx)[i].PutScalar(1.0 / mp_x[0].GlobalLength());

  // Evaluate dg/dxdot
  if (mp_dg_dxdot != NULL)
    mp_dg_dxdot->init(0.0);
  if (mp_dg_dxdotdot != NULL)
    mp_dg_dxdotdot->init(0.0);

  // Evaluate dg/dp
  if (mp_dg_dp != NULL)
    mp_dg_dp->init(0.0);
}
#endif
