//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionValuesResponseFunction.hpp"

#include "Albany_SolutionCullingStrategy.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_Assert.hpp"

#include <iostream>

#include "Albany_Application.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Thyra_VectorStdOps.hpp"

namespace Albany
{

//amb Hack for John M.'s work. Replace with an evaluator-based RF later so we
// can get gradient, etc.
//   By the way, I could be wrong, but I think NodeGIDsSolutionCullingStrategy
// isn't doing the right thing when the solution is other than scalar. But we're
// just whipping this together quickly for a demo, so I'll leave that issue (if
// indeed it is one) for later.
class SolutionValuesResponseFunction::SolutionPrinter {
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

  void print (const Teuchos::RCP<const Thyra_Vector>& g, const Teuchos::Array<GO>& eq_gids) {
    do_print(getLocalData(g), eq_gids);
  }

private:
  struct Point { ST p[3]; };

  void do_print (const Teuchos::ArrayRCP<const ST>& g,
                 const Teuchos::Array<GO>& eq_gids) {
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
      ss << filename_ << "." << app_->getComm()->getRank();
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
  void getCoordsOnRank (
    const Teuchos::Array<GO>& eq_gids, std::vector<GO>& node_gids,
    std::vector<Point>& coords, int& ndim, std::vector<std::size_t>& idxs)
  {
    Teuchos::RCP<AbstractDiscretization> disc = app_->getDiscretization();
    const Teuchos::ArrayRCP<double>& ov_coords = disc->getCoordinates();
    Teuchos::RCP<const Thyra_SpmdVectorSpace> ov_node_vs = getSpmdVectorSpace(disc->getOverlapNodeVectorSpace());
    Teuchos::RCP<const Thyra_SpmdVectorSpace> node_vs = getSpmdVectorSpace(disc->getNodeVectorSpace());
    auto ov_node_indexer = createGlobalLocalIndexer(ov_node_vs);
    auto node_indexer = createGlobalLocalIndexer(node_vs);
    ndim = disc->getNumDim();
    const int neq = disc->getNumEq();
    for (int i=0; i<eq_gids.size(); ++i) {
      const GO node_gid = eq_gids[i] / neq;
      if (!node_indexer->isLocallyOwnedElement(node_gid)) {
        continue;
      }
      idxs.push_back(i);
      node_gids.push_back(node_gid);
      const LO ov_node_lid = ov_node_indexer->getLocalElement(node_gid);
      coords.push_back(Point());
      for (int j = 0; j < ndim; ++j) {
        // 3 is used regardless of ndim.
        coords.back().p[j] = ov_coords[3*ov_node_lid + j];
      }
    }
  }
};

SolutionValuesResponseFunction::
SolutionValuesResponseFunction(const Teuchos::RCP<const Application>& app,
                               Teuchos::ParameterList& responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app),
  cullingStrategy_(createSolutionCullingStrategy(app, responseParams))
{
  sol_printer_ = SolutionPrinter::create(app_, responseParams);
}

void SolutionValuesResponseFunction::setup()
{
  cullingStrategy_->setup();
  this->updateCASManager();
}

unsigned int SolutionValuesResponseFunction::
numResponses() const
{
  if (Teuchos::nonnull(cas_manager)) {
    return getSpmdVectorSpace(cas_manager->getOverlappedVectorSpace())->localSubDim();
  }
  return 0u;
}

void SolutionValuesResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  this->updateCASManager();
  // Import the selected gids
  cas_manager->scatter(*x,*culledVec,CombineMode::INSERT);
  getNonconstLocalData(g).deepCopy(getLocalData(culledVec.getConst())());
  if (Teuchos::nonnull(sol_printer_)) {
    sol_printer_->print(g, cullingStrategy_->selectedGIDs(app_->getDisc()->getNewDOFManager()));
  }

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());

  g_->assign(*g);
}

void SolutionValuesResponseFunction::
evaluateTangent(const double /*alpha*/,
		const double beta,
		const double /* omega */,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		Teuchos::Array<ParamVec>& /*p*/,
    const int  /*parameter_index*/,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  this->updateCASManager();

  if (!g.is_null()) {
    // Import the selected gids
    cas_manager->scatter(*x,*culledVec,CombineMode::INSERT);
    getNonconstLocalData(g).deepCopy(getLocalData(culledVec.getConst())());
    if (Teuchos::nonnull(sol_printer_)) {
      sol_printer_->print(g, cullingStrategy_->selectedGIDs(app_->getDisc()->getNewDOFManager()));
    }
  }

  if (!gx.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPT(Vx.is_null());
    // Import the selected gids (only if not already done for the response)
    if (g.is_null()) {
      cas_manager->scatter(*x,*culledVec,CombineMode::INSERT);
    }
    for (int i=0; i<Vx->domain()->dim(); ++i) {
      getNonconstLocalData(gx->col(i)).deepCopy(getLocalData(culledVec.getConst())());
    }
    if (beta != 1.0) {
      gx->scale(beta);
    }
  }

  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
void SolutionValuesResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void SolutionValuesResponseFunction::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void SolutionValuesResponseFunction::
evaluate_HessVecProd_xp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void SolutionValuesResponseFunction::
evaluate_HessVecProd_px(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void SolutionValuesResponseFunction::
evaluate_HessVecProd_pp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const std::string& dist_param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void SolutionValuesResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const int  /*parameter_index*/,
		const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  this->updateCASManager();

  if (!g.is_null()) {
    // Import the selected gids
    cas_manager->scatter(*x,*culledVec,CombineMode::INSERT);
    getNonconstLocalData(g).deepCopy(getLocalData(culledVec.getConst())());
    if (Teuchos::nonnull(sol_printer_))
      sol_printer_->print(g, cullingStrategy_->selectedGIDs(app_->getDisc()->getNewDOFManager()));
  }

  if (!dg_dx.is_null()) {
    dg_dx->assign(0.0);

    auto ov_vs_indexer = createGlobalLocalIndexer(cas_manager->getOverlappedVectorSpace());
    auto deriv_vs_indexer = createGlobalLocalIndexer(dg_dx->range());
    const int colCount = dg_dx->domain()->dim();
    for (int icol = 0; icol < colCount; ++icol) {
      const GO gid = ov_vs_indexer->getGlobalElement(icol);
      const LO lid = deriv_vs_indexer->getLocalElement(gid);
      if (lid != -1) {
        auto dg_dx_localview = getNonconstLocalData(dg_dx->col(icol));
        dg_dx_localview[lid] = 1.0;
      }
    }
  }

  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void SolutionValuesResponseFunction::updateCASManager()
{
  const auto& sol_dof_mgr = app_->getDisc()->getNewDOFManager();
  const auto& solutionVS = app_->getVectorSpace();
  if (cas_manager.is_null() || !sameAs(solutionVS,cas_manager->getOwnedVectorSpace())) {
    const Teuchos::Array<GO> selectedGIDs = cullingStrategy_->selectedGIDs(sol_dof_mgr);
    Teuchos::RCP<const Thyra_VectorSpace> targetVS = createVectorSpace(app_->getComm(),selectedGIDs);

    cas_manager = createCombineAndScatterManager(solutionVS,targetVS);
    culledVec = Thyra::createMember(targetVS);
  }
}

void
SolutionValuesResponseFunction::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> out)
{
  if (g_.is_null()) {
    *out << " the response has not been evaluated yet!";
    return;
  }

  std::size_t precision = 8;
  std::size_t value_width = precision + 4;
  int gsize = g_->space()->dim();

  for (int j = 0; j < gsize; j++) {
    *out << std::setw(value_width) << Thyra::get_ele(*g_,j);
    if (j < gsize-1)
      *out << ", ";
  }
}

} // namespace Albany
