//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"
#include "Epetra_Comm.h"

Albany::SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Epetra_Comm>& comm,
                               const int numValues_) :
  SamplingBasedScalarResponseFunction(comm),
  numValues(numValues_)
{
}

Albany::SolutionValuesResponseFunction::
~SolutionValuesResponseFunction()
{
}

unsigned int
Albany::SolutionValuesResponseFunction::
numResponses() const 
{
  return numValues;
}

void
Albany::SolutionValuesResponseFunction::
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  const Epetra_BlockMap& map = x.Map();
  int N = map.NumGlobalElements();
  int max = map.MaxAllGID();

  TEUCHOS_TEST_FOR_EXCEPTION(max+1!=N, std::logic_error,
    "SolutionValuesResponse assumes contiguous GIDs. Need to ReCode!");

  std::vector<int> gids(numValues);

  int stride = 1 + (N-1)/numValues;
  for (int i=0; i<numValues; i++) gids[i] = i * stride; 

  Teuchos::RCP<Epetra_Map> valuesMap = Teuchos::rcp(new Epetra_Map(numValues, numValues, &gids[0], 0, map.Comm()) );

  Teuchos::RCP<Epetra_Vector> values = Teuchos::rcp(new Epetra_Vector(*valuesMap));
  Teuchos::RCP<Epetra_Import> valuesImport = Teuchos::rcp(new Epetra_Import(*valuesMap, map));
   
  values->Import(x, *valuesImport, Insert);
  for (int i=0; i<numValues; i++) g[i] = (*values)[i]; 
  
}

void
Albany::SolutionValuesResponseFunction::
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
  this->evaluateResponse(current_time, xdot, x, p, *g);
  cout << "SolutionValuesResponseFunction::evaluateTangent NOT IMPLEMETED" << endl;
}

void
Albany::SolutionValuesResponseFunction::
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
  //this->evaluateResponse(current_time, xdot, x, p, *g);
  cout << "SolutionValuesResponseFunction::evaluateGradient NOT IMPLEMETED" << endl;
}
