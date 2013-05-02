//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Import.h"

#include "Epetra_GatherAllV.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_Assert.hpp"

#include <algorithm>

namespace { // anonymous

void
importWithAlternateMap(
    const Epetra_Import &importer,
    const Epetra_MultiVector &source,
    Epetra_MultiVector &target,
    Epetra_CombineMode mode)
{
  const Epetra_BlockMap savedMap = target.Map();
  {
    const int ierr = target.ReplaceMap(importer.TargetMap());
    TEUCHOS_ASSERT(ierr == 0);
  }
  {
    const int ierr = target.Import(source, importer, mode);
    TEUCHOS_ASSERT(ierr == 0);
  }
  {
    const int ierr = target.ReplaceMap(savedMap);
    TEUCHOS_ASSERT(ierr == 0);
  }
}

} // anonymous namespace


Albany::SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Epetra_Comm>& comm,
                               const int numValues_) :
  SamplingBasedScalarResponseFunction(comm),
  numValues(numValues_),
  solutionImporter()
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
  this->updateSolutionImporter(x);
  importWithAlternateMap(*solutionImporter, x, g, Insert);
}

void
Albany::SolutionValuesResponseFunction::
evaluateTangent(const double /*alpha*/,
		const double beta,
		const double /*current_time*/,
		bool /*sum_derivs*/,
		const Epetra_Vector* /*xdot*/,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
		const Epetra_MultiVector* /*Vxdot*/,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* /*Vp*/,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  this->updateSolutionImporter(x);

  if (g) {
    importWithAlternateMap(*solutionImporter, x, *g, Insert);
  }

  if (gx) {
    TEUCHOS_ASSERT(Vx);
    importWithAlternateMap(*solutionImporter, *Vx, *gx, Insert);
    if (beta != 1.0) {
      gx->Scale(beta);
    }
  }

  if (gp) {
    gp->PutScalar(0.0);
  }
}


void
Albany::SolutionValuesResponseFunction::
evaluateGradient(const double /*current_time*/,
		 const Epetra_Vector* /*xdot*/,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& /*p*/,
		 ParamVec* /*deriv_p*/,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dp)
{
  this->updateSolutionImporter(x);

  if (g) {
    importWithAlternateMap(*solutionImporter, x, *g, Insert);
  }

  if (dg_dx) {
    dg_dx->PutScalar(0.0);

    const Epetra_BlockMap &replicatedMap = solutionImporter->TargetMap();
    const Epetra_BlockMap &derivMap = dg_dx->Map();
    const int colCount = dg_dx->NumVectors();
    for (int icol = 0; icol < colCount; ++icol) {
      const int lid = derivMap.LID(replicatedMap.GID(icol));
      if (lid != -1) {
        dg_dx->ReplaceMyValue(lid, icol, 1.0);
      }
    }
  }

  if (dg_dxdot) {
    dg_dxdot->PutScalar(0.0);
  }

  if (dg_dp) {
    dg_dp->PutScalar(0.0);
  }
}

void
Albany::SolutionValuesResponseFunction::
updateSolutionImporter(const Epetra_Vector& x)
{
  const Epetra_BlockMap solutionMap = x.Map();
  if (Teuchos::is_null(solutionImporter) || !solutionMap.SameAs(solutionImporter->SourceMap())) {
    solutionImporter = this->buildSolutionImporter(solutionMap);
  }
}

Teuchos::RCP<Epetra_Import>
Albany::SolutionValuesResponseFunction::
buildSolutionImporter(const Epetra_BlockMap &sourceMap)
{
  Teuchos::Array<int> allGIDs(sourceMap.NumGlobalElements());
  {
    const int ierr = Epetra::GatherAllV(
        sourceMap.Comm(),
        sourceMap.MyGlobalElements(), sourceMap.NumMyElements(),
        allGIDs.getRawPtr(), allGIDs.size());
    TEUCHOS_ASSERT(ierr == 0);
  }
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<int> selectedGIDs(numValues);
  const int stride = 1 + (allGIDs.size() - 1) / numValues;
  for (int i = 0; i < numValues; ++i) {
    selectedGIDs[i] = allGIDs[i * stride];
  }

  const Epetra_Map targetMap(numValues, numValues, selectedGIDs.getRawPtr(), 0, sourceMap.Comm());
  return Teuchos::rcp(new Epetra_Import(targetMap, sourceMap));
}
