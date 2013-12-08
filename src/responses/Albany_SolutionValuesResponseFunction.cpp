//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"

#include "Albany_SolutionCullingStrategy.hpp"

#include "Epetra_ImportWithAlternateMap.hpp"

#include "Epetra_Import.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_Assert.hpp"


Albany::SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Application>& app,
                               Teuchos::ParameterList& responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app),
  cullingStrategy_(createSolutionCullingStrategy(app, responseParams)),
  solutionImporter_()
{
}

void
Albany::SolutionValuesResponseFunction::
setup()
{
  cullingStrategy_->setup();
  this->updateSolutionImporter();
}

unsigned int
Albany::SolutionValuesResponseFunction::
numResponses() const
{
  return Teuchos::nonnull(solutionImporter_) ?
    solutionImporter_->TargetMap().NumMyElements() :
    0u;
}

void
Albany::SolutionValuesResponseFunction::
evaluateResponse(const double /*current_time*/,
		 const Epetra_Vector* /*xdot*/,
		 const Epetra_Vector* /*xdot*/,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& /*p*/,
		 Epetra_Vector& g)
{
  this->updateSolutionImporter();
  Epetra::ImportWithAlternateMap(*solutionImporter_, x, g, Insert);
}

void
Albany::SolutionValuesResponseFunction::
evaluateTangent(const double /*alpha*/,
		const double beta,
		const double omega,
		const double /*current_time*/,
		bool /*sum_derivs*/,
		const Epetra_Vector* /*xdot*/,
		const Epetra_Vector* /*xdot*/,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
		const Epetra_MultiVector* /*Vxdot*/,
		const Epetra_MultiVector* /*Vxdot*/,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* /*Vp*/,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  this->updateSolutionImporter();

  if (g) {
    Epetra::ImportWithAlternateMap(*solutionImporter_, x, *g, Insert);
  }

  if (gx) {
    TEUCHOS_ASSERT(Vx);
    Epetra::ImportWithAlternateMap(*solutionImporter_, *Vx, *gx, Insert);
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
		 const Epetra_Vector* /*xdot*/,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& /*p*/,
		 ParamVec* /*deriv_p*/,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dxdotdot,
		 Epetra_MultiVector* dg_dp)
{
  this->updateSolutionImporter();

  if (g) {
    Epetra::ImportWithAlternateMap(*solutionImporter_, x, *g, Insert);
  }

  if (dg_dx) {
    dg_dx->PutScalar(0.0);

    const Epetra_BlockMap &replicatedMap = solutionImporter_->TargetMap();
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
  if (dg_dxdotdot) {
    dg_dxdotdot->PutScalar(0.0);
  }

  if (dg_dp) {
    dg_dp->PutScalar(0.0);
  }
}

void
Albany::SolutionValuesResponseFunction::
updateSolutionImporter()
{
  const Teuchos::RCP<const Epetra_BlockMap> solutionMap = app_->getMap();
  if (Teuchos::is_null(solutionImporter_) || !solutionMap->SameAs(solutionImporter_->SourceMap())) {
    const Teuchos::Array<int> selectedGIDs = cullingStrategy_->selectedGIDs(*solutionMap);
    const Epetra_Map targetMap(-1, selectedGIDs.size(), selectedGIDs.getRawPtr(), 0, solutionMap->Comm());
    solutionImporter_ = Teuchos::rcp(new Epetra_Import(targetMap, *solutionMap));
  }
}
