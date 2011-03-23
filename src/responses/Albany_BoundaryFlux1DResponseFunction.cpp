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


#include "Albany_BoundaryFlux1DResponseFunction.hpp"

Albany::BoundaryFlux1DResponseFunction::
BoundaryFlux1DResponseFunction(unsigned int left_gid,
                               unsigned int right_gid,
                               unsigned int eqn,
                               unsigned int num_eqns,
                               double grid_spacing_,
                               const Epetra_Map& dofMap) :
  grid_spacing(grid_spacing_),
  importer(NULL),
  bv(NULL)
{
  // Compute GID's of left/right DOF's
  unsigned int left_dof = num_eqns*left_gid+eqn;
  unsigned int right_dof = num_eqns*right_gid+eqn;

  // Build importer to bring in left/right DOF's to all proc's
  int gids[4];
  gids[0] = left_dof;
  gids[1] = left_dof+num_eqns;
  gids[2] = right_dof-num_eqns;
  gids[3] = right_dof;
  boundaryMap = new Epetra_Map(4, 4, gids, 0, dofMap.Comm());
  importer = new Epetra_Import(*boundaryMap, dofMap);
  bv = new Epetra_Vector(*boundaryMap);
}

Albany::BoundaryFlux1DResponseFunction::
~BoundaryFlux1DResponseFunction()
{
  delete boundaryMap;
  delete importer;
  delete bv;
}

unsigned int
Albany::BoundaryFlux1DResponseFunction::
numResponses() const 
{
  return 2;
}

void
Albany::BoundaryFlux1DResponseFunction::
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{
  // Import boundary values
  bv->Import(x, *importer, Insert);

  // Compute fluxes
  g[0] = ((*bv)[1] - (*bv)[0]) / grid_spacing;
  g[1] = ((*bv)[3] - (*bv)[2]) / grid_spacing;
}

void
Albany::BoundaryFlux1DResponseFunction::
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
  if (g != NULL) {
    bv->Import(x, *importer, Insert);
    (*g)[0] = ((*bv)[1] - (*bv)[0]) / grid_spacing;
    (*g)[1] = ((*bv)[3] - (*bv)[2]) / grid_spacing;
  }

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<gt.size(); j++)
    if (gt[j] != Teuchos::null) {
      Epetra_MultiVector bvt(*boundaryMap, dx_dp[j]->NumVectors());
      bvt.Import(*dx_dp[j], *importer, Insert);
      for (int i=0; i<dx_dp[j]->NumVectors(); i++) {
	(*gt[j])[i][0] = (bvt[i][1] - bvt[i][0]) / grid_spacing;
	(*gt[j])[i][1] = (bvt[i][3] - bvt[i][2]) / grid_spacing;
      }
    }
}

void
Albany::BoundaryFlux1DResponseFunction::
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
  if (g != NULL) {
    bv->Import(x, *importer, Insert);
    (*g)[0] = ((*bv)[1] - (*bv)[0]) / grid_spacing;
    (*g)[1] = ((*bv)[3] - (*bv)[2]) / grid_spacing;
  }

  // Evaluate dg/dx
  if (dg_dx != NULL) {
    Epetra_MultiVector bv_dx(*boundaryMap, 2);
    bv_dx.PutScalar(0.0);
    bv_dx[0][0] = -1.0 / grid_spacing;
    bv_dx[0][1] =  1.0 / grid_spacing;
    bv_dx[1][2] = -1.0 / grid_spacing;
    bv_dx[1][3] =  1.0 / grid_spacing;
    dg_dx->PutScalar(0.0);
    dg_dx->Export(bv_dx, *importer, Insert);
  }

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);

  // Evaluate dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<dg_dp.size(); j++)
    if (dg_dp[j] != Teuchos::null)
      dg_dp[j]->PutScalar(0.0);
}

void
Albany::BoundaryFlux1DResponseFunction::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{
  unsigned int sz = sg_x.size();
  for (unsigned int i=0; i<sz; i++) {

    // Import boundary values
    bv->Import(sg_x[i], *importer, Insert);
    
    // Compute fluxes
    sg_g[i][0] = ((*bv)[1] - (*bv)[0]) / grid_spacing;
    sg_g[i][1] = ((*bv)[3] - (*bv)[2]) / grid_spacing;

  }
}
