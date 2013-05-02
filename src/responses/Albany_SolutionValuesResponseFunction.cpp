//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"

#include "Epetra_Comm.h"

#include "Epetra_GatherAllV.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"

#include <algorithm>

namespace { // anonymous

Teuchos::RCP<Epetra_Import>
buildReplicationImporter(const Epetra_BlockMap &distributedMap, const int numValues)
{
  Teuchos::Array<int> allGIDs(distributedMap.NumGlobalElements());
  Epetra::GatherAllV(
      distributedMap.Comm(),
      distributedMap.MyGlobalElements(), distributedMap.NumMyElements(),
      allGIDs.getRawPtr(), allGIDs.size());
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<int> selectedGIDs(numValues);
  const int stride = 1 + (allGIDs.size() - 1) / numValues;
  for (int i = 0; i < numValues; ++i) {
    selectedGIDs[i] = allGIDs[i * stride];
  }

  const Epetra_Map replicatedCulledMap(numValues, numValues, selectedGIDs.getRawPtr(), 0, distributedMap.Comm());
  return Teuchos::rcp(new Epetra_Import(replicatedCulledMap, distributedMap));
}

} // anonymous namespace


Albany::SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Epetra_Comm>& comm,
                               const int numValues_) :
  SamplingBasedScalarResponseFunction(comm),
  numValues(numValues_)
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
evaluateResponse(const double /*current_time*/,
		 const Epetra_Vector* /*xdot*/,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& /*p*/,
		 Epetra_Vector& g)
{
  const Teuchos::RCP<const Epetra_Import> valuesImporter =
    buildReplicationImporter(x.Map(), numValues);

  Epetra_Vector replicatedCulledX(valuesImporter->TargetMap(), /* zeroOut = */ false);
  replicatedCulledX.Import(x, *valuesImporter, Insert);

  std::copy(replicatedCulledX.Values(), replicatedCulledX.Values() + numValues, g.Values());
}

void
Albany::SolutionValuesResponseFunction::
evaluateTangent(const double /*alpha*/,
		const double /*beta*/,
		const double current_time,
		bool /*sum_derivs*/,
		const Epetra_Vector* xdot,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* /*deriv_p*/,
		const Epetra_MultiVector* /*Vxdot*/,
		const Epetra_MultiVector* /*Vx*/,
		const Epetra_MultiVector* /*Vp*/,
		Epetra_Vector* g,
		Epetra_MultiVector* /*gx*/,
		Epetra_MultiVector* /*gp*/)
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
		 ParamVec* /*deriv_p*/,
		 Epetra_Vector* g,
		 Epetra_MultiVector* /*dg_dx*/,
		 Epetra_MultiVector* /*dg_dxdot*/,
		 Epetra_MultiVector* /*dg_dp*/)
{
  this->evaluateResponse(current_time, xdot, x, p, *g);
  cout << "SolutionValuesResponseFunction::evaluateGradient NOT IMPLEMETED" << endl;
}
