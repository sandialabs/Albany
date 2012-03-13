/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Albany_SolutionAverageResponseFunction.hpp"

Albany::SolutionAverageResponseFunction::
SolutionAverageResponseFunction(const Teuchos::RCP<const Epetra_Comm>& comm) :
  ScalarResponseFunction(comm)
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
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  x.MeanValue(&g[0]);
}

void
Albany::SolutionAverageResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double current_time,
		bool sum_derivs,
		const Epetra_Vector* xdot,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Epetra_MultiVector* Vxdot,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* Vp,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  // Evaluate response g
  if (g != NULL)
    x.MeanValue(&(*g)[0]);

  // Evaluate tangent of g = dg/dx*Vx + dg/dxdot*Vxdot + dg/dp*Vp
  // If Vx == NULL, Vx is the identity
  if (gx != NULL) {
    if (Vx != NULL)
      for (int j=0; j<Vx->NumVectors(); j++)
	(*Vx)(j)->MeanValue(&(*gx)[j][0]);
    else
      gx->PutScalar(1.0/x.GlobalLength());
    gx->Scale(alpha);
  }
  if (gp != NULL)
    gp->PutScalar(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateGradient(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
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

  // Evaluate dg/dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateSGResponse(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
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
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
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
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
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

  // Evaluate dg/dp
  if (sg_dg_dp != NULL)
    sg_dg_dp->init(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateMPResponse(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
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
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
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
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_dg_dx,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
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

  // Evaluate dg/dp
  if (mp_dg_dp != NULL)
    mp_dg_dp->init(0.0);
}
