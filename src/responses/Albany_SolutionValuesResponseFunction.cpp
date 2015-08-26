//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"

#include "Albany_SolutionCullingStrategy.hpp"

#if defined(ALBANY_EPETRA)
//#include "Epetra_ImportWithAlternateMap.hpp"
#include "Epetra_Import.h"
#endif

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_Assert.hpp"

#include <iostream>

//amb Hack for John M.'s work. Replace with an evaluator-based RF later so we
// can get gradient, etc.
//   By the way, I could be wrong, but I think NodeGIDsSolutionCullingStrategy
// isn't doing the right thing when the solution is other than scalar. But we're
// just whipping this together quickly for a demo, so I'll leave that issue (if
// indeed it is one) for later.
class Albany::SolutionValuesResponseFunction::SolutionPrinter {
private:
  std::string filename_;
  // No need for a separate RCP.
  const Teuchos::RCP<const Application>& app_;

public:
  SolutionPrinter (const Teuchos::RCP<const Application>& app,
                   Teuchos::ParameterList& response_parms)
    : app_(app)
  {
    filename_ = response_parms.get<std::string>("Output File");
  }

  static Teuchos::RCP<SolutionPrinter> create (
    const Teuchos::RCP<const Application>& app,
    Teuchos::ParameterList& response_parms)
  {
    if (response_parms.isType<std::string>("Output File"))
      return Teuchos::rcp(new SolutionPrinter(app, response_parms));
    return Teuchos::null;
  }

#if defined(ALBANY_EPETRA)
  void print (const Epetra_Vector& g, const Teuchos::Array<int>& eq_gids) {
    print(Teuchos::arrayView(g.Values(), g.MyLength()), eq_gids);
  }
#endif

  void print (const Tpetra_Vector& g, const Teuchos::Array<GO>& eq_gids) {
    print(Teuchos::arrayView(&g.get1dView()[0], g.getLocalLength()), eq_gids);
  }

private:
  struct Point { ST p[3]; };

  template<typename gid_type>
  void print (const Teuchos::ArrayView<const ST>& g,
              const Teuchos::Array<gid_type>& eq_gids) {
    TEUCHOS_TEST_FOR_EXCEPTION(g.size() != eq_gids.size(), std::logic_error,
                               "g.size() != eq_gids.size()");

    // Get all coordinates.
    std::vector<GO> node_gids;
    std::vector<Point> coords;
    int ndim;
    std::vector<std::size_t> idxs;
    getCoordsOnRank(eq_gids, node_gids, coords, ndim, idxs);

    std::string filename; {
      std::stringstream ss;
      ss << filename_ << "." << app_->getMapT()->getComm()->getRank();
      filename = ss.str();
    }

    std::fstream out;
    out.open(filename.c_str(), std::fstream::out);
    out.precision(15);
    for (std::size_t i = 0; i < node_gids.size(); ++i) {
      // Generally the node gid for the user starts at 1.
      out << std::setw(11) << node_gids[i] + 1;
      out << std::scientific;
      for (int j = 0; j < ndim; ++j)
        out << " " << std::setw(22) << coords[i].p[j];
      out << " " << std::setw(22) << g[idxs[i]] << std::endl;
    }
    out.close();
  }

  // gids is global equation ids.
  template<typename gid_type>
  void getCoordsOnRank (
    const Teuchos::Array<gid_type>& eq_gids, std::vector<GO>& node_gids,
    std::vector<Point>& coords, int& ndim, std::vector<std::size_t>& idxs)
  {
    Teuchos::RCP<Albany::AbstractDiscretization> d = app_->getDiscretization();
    const Teuchos::ArrayRCP<double>& ol_coords = d->getCoordinates();
    Teuchos::RCP<const Tpetra_Map>
      ol_node_map = d->getOverlapNodeMapT(),
      node_map = d->getNodeMapT();
    ndim = d->getNumDim();
    const int neq = d->getNumEq();
    for (std::size_t i = 0; i < eq_gids.size(); ++i) {
      const GO node_gid = eq_gids[i] / neq;
      if ( ! node_map->isNodeGlobalElement(node_gid)) continue;
      idxs.push_back(i);
      node_gids.push_back(node_gid);
      const LO ol_node_lid = ol_node_map->getLocalElement(node_gid);
      coords.push_back(Point());
      for (int j = 0; j < ndim; ++j) {
        // 3 is used regardless of ndim.
        coords.back().p[j] = ol_coords[3*ol_node_lid + j];
      }
    }
  }
};

Albany::SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Application>& app,
                               Teuchos::ParameterList& responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app),
  cullingStrategy_(createSolutionCullingStrategy(app, responseParams))
{
  sol_printer_ = SolutionPrinter::create(app_, responseParams);
}

#if defined(ALBANY_EPETRA)
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
#if defined(ALBANY_EPETRA)
  if (Teuchos::nonnull(solutionImporter_))
    return solutionImporter_->TargetMap().NumMyElements();
#endif
  return 0u;
}

#if defined(ALBANY_EPETRA)
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
  if (Teuchos::nonnull(sol_printer_))
    sol_printer_->print(g, cullingStrategy_->selectedGIDs(*app_->getMap()));
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
  if (Teuchos::nonnull(sol_printer_))
    sol_printer_->print(gT, cullingStrategy_->selectedGIDsT(app_->getMapT()));
}

#if defined(ALBANY_EPETRA)
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
    if (Teuchos::nonnull(sol_printer_))
      sol_printer_->print(*g, cullingStrategy_->selectedGIDs(*app_->getMap()));
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
    if (Teuchos::nonnull(sol_printer_))
      sol_printer_->print(*gT, cullingStrategy_->selectedGIDsT(app_->getMapT()));
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


#if defined(ALBANY_EPETRA)
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
    if (Teuchos::nonnull(sol_printer_))
      sol_printer_->print(*g, cullingStrategy_->selectedGIDs(*app_->getMap()));
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

#if defined(ALBANY_EPETRA)
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
    if (Teuchos::nonnull(sol_printer_))
      sol_printer_->print(*gT, cullingStrategy_->selectedGIDsT(app_->getMapT()));
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


#if defined(ALBANY_EPETRA)
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
#if defined(ALBANY_EPETRA)
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

