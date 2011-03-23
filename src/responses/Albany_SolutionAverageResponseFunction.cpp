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
SolutionAverageResponseFunction()
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
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{
  x.MeanValue(&g[0]);
}

void
Albany::SolutionAverageResponseFunction::
evaluateTangents(
	   const Epetra_Vector* xdot,
	   const Epetra_Vector& x,
	   const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	   const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dxdot_dp,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dx_dp,
	   Epetra_Vector* g,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& gt)
{
  // Evaluate response g
  if (g != NULL)
    x.MeanValue(&(*g)[0]);

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<gt.size(); j++)
    if (gt[j] != Teuchos::null)
      for (int i=0; i<dx_dp[i]->NumVectors(); i++)
	(*dx_dp[j])(i)->MeanValue(&(*gt[j])[i][0]);
}

void
Albany::SolutionAverageResponseFunction::
evaluateGradients(
	  const Epetra_Vector* xdot,
	  const Epetra_Vector& x,
	  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	  const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	  Epetra_Vector* g,
	  Epetra_MultiVector* dg_dx,
	  Epetra_MultiVector* dg_dxdot,
	  const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dg_dp)
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
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<dg_dp.size(); j++)
    if (dg_dp[j] != Teuchos::null)
      dg_dp[j]->PutScalar(0.0);
}

void
Albany::SolutionAverageResponseFunction::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{
  unsigned int sz = sg_x.size();
  for (unsigned int i=0; i<sz; i++)
    sg_x[i].MeanValue(&sg_g[i][0]);
}
