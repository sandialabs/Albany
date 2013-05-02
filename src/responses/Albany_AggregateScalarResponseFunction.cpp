//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AggregateScalarResponseFunction.hpp"
#include "Epetra_LocalMap.h"

using Teuchos::RCP;
using Teuchos::rcp;

Albany::AggregateScalarResponseFunction::
AggregateScalarResponseFunction(
  const Teuchos::RCP<const Epetra_Comm>& comm,
  const Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> >& responses_) :
  SamplingBasedScalarResponseFunction(comm),
  responses(responses_)
{
}

void
Albany::AggregateScalarResponseFunction::
setup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction> > ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->setup();
  }
}

Albany::AggregateScalarResponseFunction::
~AggregateScalarResponseFunction()
{
}

unsigned int
Albany::AggregateScalarResponseFunction::
numResponses() const 
{
  unsigned int n = 0;
  for (int i=0; i<responses.size(); i++)
    n += responses[i]->numResponses();
  return n;
}

void
Albany::AggregateScalarResponseFunction::
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, 
				       *(responses[i]->getComm()));

    // Create Epetra_Vector for response function
    Epetra_Vector local_g(local_response_map);

    // Evaluate response function
    responses[i]->evaluateResponse(current_time, xdot, x, p, local_g);
    
    // Copy result into combined result
    for (unsigned int j=0; j<num_responses; j++)
      g[offset+j] = local_g[j];

    // Increment offset in combined result
    offset += num_responses;
  }
}

void
Albany::AggregateScalarResponseFunction::
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
  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, 
      *(responses[i]->getComm()));

    // Create Epetra_Vectors for response function
    RCP<Epetra_Vector> local_g;
    RCP<Epetra_MultiVector> local_gx, local_gp;
    if (g != NULL)
      local_g = rcp(new Epetra_Vector(local_response_map));
    if (gx != NULL)
      local_gx = rcp(new Epetra_MultiVector(local_response_map, 
					    gx->NumVectors()));
    if (gp != NULL)
      local_gp = rcp(new Epetra_MultiVector(local_response_map, 
					    gp->NumVectors()));

    // Evaluate response function
    responses[i]->evaluateTangent(alpha, beta, current_time, sum_derivs,
				  xdot, x, p, deriv_p, Vxdot, Vx, Vp, 
				  local_g.get(), local_gx.get(), 
				  local_gp.get());

    // Copy results into combined result
    for (unsigned int j=0; j<num_responses; j++) {
      if (g != NULL)
        (*g)[offset+j] = (*local_g)[j];
      if (gx != NULL)
	for (int k=0; k<gx->NumVectors(); k++)
	  (*gx)[k][offset+j] = (*local_gx)[k][j];
      if (gp != NULL)
	for (int k=0; k<gp->NumVectors(); k++)
	  (*gp)[k][offset+j] = (*local_gp)[k][j];
    }

    // Increment offset in combined result
    offset += num_responses;
  }
}

void
Albany::AggregateScalarResponseFunction::
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
  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {

    // Create Epetra_Map for response function
    int num_responses = responses[i]->numResponses();
    Epetra_LocalMap local_response_map(num_responses, 0, 
				       *(responses[i]->getComm()));

    // Create Epetra_Vectors for response function
    RCP<Epetra_Vector> local_g;
    if (g != NULL)
      local_g = rcp(new Epetra_Vector(local_response_map));
    RCP<Epetra_MultiVector> local_dgdx;
    if (dg_dx != NULL)
      local_dgdx = rcp(new Epetra_MultiVector(dg_dx->Map(), num_responses));
    RCP<Epetra_MultiVector> local_dgdxdot;
    if (dg_dxdot != NULL)
      local_dgdxdot = rcp(new Epetra_MultiVector(dg_dxdot->Map(), 
						 num_responses));
    RCP<Epetra_MultiVector> local_dgdp;
    if (dg_dp != NULL)
      local_dgdp = rcp(new Epetra_MultiVector(local_response_map, 
					      dg_dp->NumVectors()));

    // Evaluate response function
    responses[i]->evaluateGradient(current_time, xdot, x, p, deriv_p, 
				   local_g.get(), local_dgdx.get(), 
				   local_dgdxdot.get(), local_dgdp.get());

    // Copy results into combined result
    for (unsigned int j=0; j<num_responses; j++) {
      if (g != NULL)
        (*g)[offset+j] = (*local_g)[j];
      if (dg_dx != NULL)
        (*dg_dx)(offset+j)->Update(1.0, *((*local_dgdx)(j)), 0.0);
      if (dg_dxdot != NULL)
        (*dg_dxdot)(offset+j)->Update(1.0, *((*local_dgdxdot)(j)), 0.0);
      if (dg_dp != NULL)
	for (int k=0; k<dg_dp->NumVectors(); k++)
	  (*dg_dp)[k][offset+j] = (*local_dgdp)[k][j];
    }

    // Increment offset in combined result
    offset += num_responses;
  }
}
