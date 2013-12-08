//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Epetra_Comm.h"

Albany::SolutionMaxValueResponseFunction::
SolutionMaxValueResponseFunction(const Teuchos::RCP<const Epetra_Comm>& comm,
				 int neq_, int eq_, bool interleavedOrdering_) :
  SamplingBasedScalarResponseFunction(comm),
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
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector* xdotdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  int index;
  computeMaxValue(x, g[0], index);
}

void
Albany::SolutionMaxValueResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
		const Epetra_Vector* xdot,
		const Epetra_Vector* xdotdot,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Epetra_MultiVector* Vxdot,
		const Epetra_MultiVector* Vxdotdot,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* Vp,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  Teuchos::RCP<Epetra_MultiVector> dgdx;
  if (gx != NULL && Vx != NULL)
    dgdx = Teuchos::rcp(new Epetra_MultiVector(x.Map(), 1));
  else
    dgdx = Teuchos::rcp(gx,false);
  evaluateGradient(current_time, xdot, xdotdot, x, p, deriv_p, g, dgdx.get(), NULL, NULL, gp);
  if (gx != NULL && Vx != NULL)
    gx->Multiply('T', 'N', alpha, *dgdx, *Vx, 0.0);
}

void
Albany::SolutionMaxValueResponseFunction::
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
