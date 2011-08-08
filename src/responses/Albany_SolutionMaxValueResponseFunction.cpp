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


#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Epetra_Comm.h"

Albany::SolutionMaxValueResponseFunction::
SolutionMaxValueResponseFunction(int neq_, int eq_, bool interleavedOrdering_) :
  neq(neq_), eq(eq_), interleavedOrdering(interleavedOrdering_)
{
}

Albany::SolutionMaxValueResponseFunction::
~SolutionMaxValueResponseFunction()
{
}

unsigned int
Albany::SolutionMaxValueResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::SolutionMaxValueResponseFunction::
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{
  int index;
  computeMaxValue(x, g[0], index);
}

void
Albany::SolutionMaxValueResponseFunction::
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
  throw 54;
}

void
Albany::SolutionMaxValueResponseFunction::
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
  int global_index;
  double mxv;
  computeMaxValue(x, mxv, global_index);

  // Evaluate response g
  if (g != NULL)
    (*g)[0] = mxv;

  // Evaluate dg/dx
  if (dg_dx != NULL) {
    int im = -1;
    for (int i=0; i<x.Map().NumMyElements(); i++) {
       if (x[i] == mxv) { (*dg_dx)[0][i] = 1.0; im = i; }
       else             (*dg_dx)[0][i] = 0.0;
    }

//DEBUG 3 lines
    double nrm;
    (*dg_dx)(0)->Norm1(&nrm);
    cout << " Max Value Response: grad norm1 =  " << nrm << "  at unk# " << im << endl;

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
Albany::SolutionMaxValueResponseFunction::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{

  // find node im with max value
  int global_index;
  double mxv;
  computeMaxValue(sg_x[0], mxv, global_index);

  // copy its coeffs to g
  if (sg_x[0].Map().MyGID(global_index)) {
    int im = sg_x[0].Map().LID(global_index);
    unsigned int sz = sg_x.size();
    for (unsigned int i=0; i<sz; i++)
      sg_g[i][0] = sg_x[i][im];
  }
  else {
    // We need to do communication in the parallel case
    throw 54;
  }

}

void
Albany::SolutionMaxValueResponseFunction::
computeMaxValue(const Epetra_Vector& x, double& global_max, int& global_index)
{
  double my_max = -Epetra_MaxDouble;
  int my_index = -1, index;
  
  // Loop over nodes to find max value for equation eq
  int num_my_nodes = x.MyLength() / neq;
  for (int node=0; node<num_my_nodes; node++) {
    if (interleavedOrdering)  index = node*neq+eq;
    else                      index = node + eq*num_my_nodes;
    if (x[index] > my_max) {
      my_max = x[index];
      my_index = index;
    }
  }

  // Check remainder (AGS: NOT SURE HOW THIS CODE GETS CALLED?)
  if (num_my_nodes*neq+eq < x.MyLength()) {
    if (interleavedOrdering)  index = num_my_nodes*neq+eq;
    else                      index = num_my_nodes + eq*num_my_nodes;
    if (x[index] > my_max) {
      my_max = x[index];
      my_index = index;
    }
  }

  // Get max value across all proc's
  x.Comm().MaxAll(&my_max, &global_max, 1);

  // Compute min of all global indices equal to max value
  if (my_max == global_max)
    my_index = x.Map().GID(my_index);
  else
    my_index = x.GlobalLength();
  x.Comm().MinAll(&my_index, &global_index, 1);
}
