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


#include "Albany_EvaluatedResponseFunction.hpp"

Albany::EvaluatedResponseFunction::
EvaluatedResponseFunction()
{
}

Albany::EvaluatedResponseFunction::
~EvaluatedResponseFunction()
{
}

unsigned int
Albany::EvaluatedResponseFunction::
numResponses() const 
{
  return responseInitVals.size();
}

void
Albany::EvaluatedResponseFunction::
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{
  for(unsigned int i=0; i < responseInitVals.size(); ++i)
    g[i] = responseInitVals[i];
}

void
Albany::EvaluatedResponseFunction::
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
    for(unsigned int i=0; i < responseInitVals.size(); ++i)
      (*g)[i] = responseInitVals[i];
  }

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<gt.size(); j++)
    if (gt[j] != Teuchos::null)
      for (int i=0; i<dx_dp[i]->NumVectors(); i++)
	(*gt[j])[i][0] = 0.0;
}

void
Albany::EvaluatedResponseFunction::
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
    for(unsigned int i=0; i < responseInitVals.size(); ++i)
      (*g)[i] = responseInitVals[i];
  }

  // Evaluate dg/dx
  if (dg_dx != NULL)
    dg_dx->PutScalar(0.0);

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);

  // Evaluate dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<dg_dp.size(); j++)
    if (dg_dp[j] != Teuchos::null)
      dg_dp[j]->PutScalar(0.0);
}

void
Albany::EvaluatedResponseFunction::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{
  unsigned int sz = sg_x.size();
  for (unsigned int i=0; i<sz; i++)
    sg_g[i][0] = 0.0;
}



void 
Albany::EvaluatedResponseFunction::
postProcessResponses(const Epetra_Comm& comm, Teuchos::RCP<Epetra_Vector>& g)
{
  std::string type = postProcessingParams.get<std::string>("Processing Type");

  if( type == "Sum" ) {
    comm.SumAll(g->Values(), g->Values(), g->MyLength());
  }
  else if( type == "Min" ) {
    int indexToMin = postProcessingParams.get<int>("Index");
    double min;
    comm.MinAll( &((*g)[indexToMin]), &min, 1);
    
    int procToBcast;
    if( (*g)[indexToMin] == min ) 
      procToBcast = comm.MyPID();
    else procToBcast = -1;

    int winner;
    comm.MaxAll(&procToBcast, &winner, 1);
    comm.Broadcast( g->Values(), g->MyLength(), winner);
  }
  else if( type == "Max") {
    int indexToMax = postProcessingParams.get<int>("Index");
    double max;
    comm.MaxAll(&((*g)[indexToMax]), &max, 1);
    
    int procToBcast;
    if( (*g)[indexToMax] == max ) 
      procToBcast = comm.MyPID();
    else procToBcast = -1;

    int winner;
    comm.MaxAll(&procToBcast, &winner, 1);
    comm.Broadcast( g->Values(), g->MyLength(), winner);
  }
  else if( type == "None") {
  }
  else TEST_FOR_EXCEPT(true);
}

void 
Albany::EvaluatedResponseFunction::
postProcessResponseDerivatives(const Epetra_Comm& comm, Teuchos::RCP<Epetra_MultiVector>& gt)
{
  //TODO - but maybe there's nothing to do here, since derivative is local to processors?
}

void 
Albany::EvaluatedResponseFunction::
setResponseInitialValues(const std::vector<double>& initVals)
{
  responseInitVals = initVals;
}
 
void 
Albany::EvaluatedResponseFunction::
setResponseInitialValues(double singleInitValForAll, unsigned int numberOfResponses)
{
  responseInitVals.resize(numberOfResponses);
  for(unsigned int i=0; i < numberOfResponses; ++i)
    responseInitVals[i] = singleInitValForAll;
}


void 
Albany::EvaluatedResponseFunction::
setPostProcessingParams(const Teuchos::ParameterList& params)
{ 
  postProcessingParams = params;  
}
