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
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  for(unsigned int i=0; i < responseInitVals.size(); ++i)
    g[i] = responseInitVals[i];
}

void
Albany::EvaluatedResponseFunction::
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
  if (g != NULL) {
    for(unsigned int i=0; i < responseInitVals.size(); ++i)
      (*g)[i] = responseInitVals[i];
  }

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  if (gx != NULL)
    gx->PutScalar(0.0);
  if (gp != NULL)
    gp->PutScalar(0.0);
}

void
Albany::EvaluatedResponseFunction::
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

  // Evaluate dg/dxdot
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}

void 
Albany::EvaluatedResponseFunction::
postProcessResponses(const Epetra_Comm& comm, Teuchos::RCP<Epetra_Vector>& g)
{
  std::string type = postProcessingParams.get<std::string>("Processing Type");

  if( type == "Sum" ) {  //NOTE: assumes MyLength == 1 (SumAll only works then)
    bool hugeIfNonPos = postProcessingParams.get<bool>("Huge if non-positive", false);
    comm.SumAll(g->Values(), g->Values(), g->MyLength()); 
    
    if(hugeIfNonPos && (*g)[0] < 1e-6) {
      (*g)[0] = 1e+100;
    }
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
  else if( type == "SumThenNormalize" ) {
    std::size_t len = g->MyLength();
    double* summed = new double[len];

    comm.SumAll(g->Values(), summed, len);    
    for(std::size_t i=0; i<len; i++) (*g)[i] = summed[i];
    delete [] summed;

    int iNormalizer = postProcessingParams.get<int>("Normalizer Index");

    if( fabs((*g)[iNormalizer]) > 1e-9 ) {
      for( int i=0; i < g->MyLength(); i++) {
	if( i == iNormalizer ) continue;
	(*g)[i] = (*g)[i] / (*g)[iNormalizer];
      }
      (*g)[iNormalizer] = 1.0;
    }
  }
  else if( type == "None") {
  }
  else TEUCHOS_TEST_FOR_EXCEPT(true);
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
