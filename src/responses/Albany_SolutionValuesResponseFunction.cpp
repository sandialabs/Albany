//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"

#include "Albany_SolutionCullingStrategy.hpp"

#ifdef ALBANY_EPETRA
//#include "Epetra_ImportWithAlternateMap.hpp"
#include "Epetra_Import.h"
#endif

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_Assert.hpp"

#include <iostream>

Albany::SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Application>& app,
                               Teuchos::ParameterList& responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app),
  cullingStrategy_(createSolutionCullingStrategy(app, responseParams))
{
}

#ifdef ALBANY_EPETRA
void
Albany::SolutionValuesResponseFunction::
setup()
{
  cullingStrategy_->setup();
  this->updateSolutionImporter();
}
#endif

void
Albany::SolutionValuesResponseFunction::
setupT()
{
  cullingStrategy_->setupT();
  this->updateSolutionImporterT();
}

unsigned int
Albany::SolutionValuesResponseFunction::
numResponses() const
{
  if (Teuchos::nonnull(solutionImporterT_))
    return solutionImporterT_->getTargetMap()->getNodeNumElements();
#ifdef ALBANY_EPETRA
  if (Teuchos::nonnull(solutionImporter_))
    return solutionImporter_->TargetMap().NumMyElements();
#endif
  return 0u;
}

#ifdef ALBANY_EPETRA
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
  this->ImportWithAlternateMap(*solutionImporter_, x, g, Insert);
}
#endif

void
Albany::SolutionValuesResponseFunction::
evaluateResponseT(const double /*current_time*/,
		 const Tpetra_Vector* /*xdot*/,
		 const Tpetra_Vector* /*xdot*/,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& /*p*/,
		 Tpetra_Vector& gT)
{
  this->updateSolutionImporterT();
  this->ImportWithAlternateMapT(solutionImporterT_, xT, gT, Tpetra::INSERT);
}

#ifdef ALBANY_EPETRA
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
    this->ImportWithAlternateMap(*solutionImporter_, x, *g, Insert);
  }

  if (gx) {
    TEUCHOS_ASSERT(Vx);
    this->ImportWithAlternateMap(*solutionImporter_, *Vx, *gx, Insert);
    if (beta != 1.0) {
      gx->Scale(beta);
    }
  }

  if (gp) {
    gp->PutScalar(0.0);
  }
}
#endif


void
Albany::SolutionValuesResponseFunction::
evaluateTangentT(const double /*alpha*/,
		const double beta,
		const double omega,
		const double /*current_time*/,
		bool /*sum_derivs*/,
		const Tpetra_Vector* /*xdot*/,
		const Tpetra_Vector* /*xdot*/,
		const Tpetra_Vector& xT,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
		const Tpetra_MultiVector* /*Vxdot*/,
		const Tpetra_MultiVector* /*Vxdot*/,
		const Tpetra_MultiVector* VxT,
		const Tpetra_MultiVector* /*Vp*/,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{
  this->updateSolutionImporterT();

  if (gT) {
    this->ImportWithAlternateMapT(solutionImporterT_, xT, *gT, Tpetra::INSERT);
  }

  if (gxT) {
    TEUCHOS_ASSERT(VxT);
    this->ImportWithAlternateMapT(solutionImporterT_, *VxT, gxT, Tpetra::INSERT);
    if (beta != 1.0) {
      gxT->scale(beta);
    }
  }

  if (gpT) {
    gpT->putScalar(0.0);
  }
}


#ifdef ALBANY_EPETRA
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
    this->ImportWithAlternateMap(*solutionImporter_, x, *g, Insert);
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
#endif

#ifdef ALBANY_EPETRA
//! Evaluate distributed parameter derivative dg/dp
void
Albany::SolutionValuesResponseFunction::
evaluateDistParamDeriv(
    const double current_time,
    const Epetra_Vector* xdot,
    const Epetra_Vector* xdotdot,
    const Epetra_Vector& x,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Epetra_MultiVector* dg_dp)
{
  if (dg_dp) {
      dg_dp->PutScalar(0.0);
  }
}
#endif // ALBANY_EPETRA

void
Albany::SolutionValuesResponseFunction::
evaluateGradientT(const double /*current_time*/,
		 const Tpetra_Vector* /*xdot*/,
		 const Tpetra_Vector* /*xdot*/,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& /*p*/,
		 ParamVec* /*deriv_p*/,
		 Tpetra_Vector* gT,
		 Tpetra_MultiVector* dg_dxT,
		 Tpetra_MultiVector* dg_dxdotT,
		 Tpetra_MultiVector* dg_dxdotdotT,
		 Tpetra_MultiVector* dg_dpT)
{
  this->updateSolutionImporterT();

  if (gT) {
    this->ImportWithAlternateMapT(solutionImporterT_, xT, *gT, Tpetra::INSERT);
  }

  if (dg_dxT) {
    dg_dxT->putScalar(0.0);

    Teuchos::RCP<const Tpetra_Map> replicatedMapT = solutionImporterT_->getTargetMap();
    Teuchos::RCP<const Tpetra_Map> derivMapT = dg_dxT->getMap();
    const int colCount = dg_dxT->getNumVectors();
    for (int icol = 0; icol < colCount; ++icol) {
      const int lid = derivMapT->getLocalElement(replicatedMapT->getGlobalElement(icol));
      if (lid != -1) {
        dg_dxT->replaceLocalValue(lid, icol, 1.0);
      }
    }
  }

  if (dg_dxdotT) {
    dg_dxdotT->putScalar(0.0);
  }
  if (dg_dxdotdotT) {
    dg_dxdotdotT->putScalar(0.0);
  }

  if (dg_dpT) {
    dg_dpT->putScalar(0.0);
  }
}


#ifdef ALBANY_EPETRA
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
#endif

void
Albany::SolutionValuesResponseFunction::
updateSolutionImporterT()
{
  const Teuchos::RCP<const Tpetra_Map> solutionMapT = app_->getMapT();
  if (Teuchos::is_null(solutionImporterT_) || !solutionMapT->isSameAs(*solutionImporterT_->getSourceMap())) {
    const Teuchos::Array<GO> selectedGIDsT = cullingStrategy_->selectedGIDsT(solutionMapT);
    Teuchos::RCP<const Tpetra_Map> targetMapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (selectedGIDsT, solutionMapT->getComm(), solutionMapT->getNode());
    //const Epetra_Map targetMap(-1, selectedGIDs.size(), selectedGIDs.getRawPtr(), 0, solutionMap->Comm());
    solutionImporterT_ = Teuchos::rcp(new Tpetra_Import(solutionMapT, targetMapT));
  }
}
#ifdef ALBANY_EPETRA
void
Albany::SolutionValuesResponseFunction::
ImportWithAlternateMap(
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
#endif

void
Albany::SolutionValuesResponseFunction::
ImportWithAlternateMapT(
    Teuchos::RCP<const Tpetra_Import> importerT,
    const Tpetra_MultiVector& sourceT,
    Tpetra_MultiVector* targetT,
    Tpetra::CombineMode modeT)
{
  Teuchos::RCP<const Tpetra_Map> savedMapT = targetT->getMap();
  targetT->replaceMap(importerT->getTargetMap());
  targetT->doImport(sourceT, *importerT, modeT);
  targetT->replaceMap(savedMapT);
}

void
Albany::SolutionValuesResponseFunction::
ImportWithAlternateMapT(
    Teuchos::RCP<const Tpetra_Import> importerT,
    const Tpetra_Vector& sourceT,
    Tpetra_Vector& targetT,
    Tpetra::CombineMode modeT)
{
  Teuchos::RCP<const Tpetra_Map> savedMapT = targetT.getMap();
  targetT.replaceMap(importerT->getTargetMap());
  targetT.doImport(sourceT, *importerT, modeT);
  targetT.replaceMap(savedMapT);
}

