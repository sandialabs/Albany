//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Thyra_SpmdVectorBase.hpp"
#include "Thyra_VectorStdOps.hpp"

#include <limits>

namespace Albany
{

SolutionMaxValueResponseFunction::
SolutionMaxValueResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& comm,
                                  int neq_, int eq_, DiscType interleavedOrdering_)
 : SamplingBasedScalarResponseFunction(comm)
 , neq(neq_)
 , eq(eq_)
 , comm_(comm)
 , interleavedOrdering(interleavedOrdering_)
{
  // Nothing to be done here
}

void SolutionMaxValueResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  Teuchos::ArrayRCP<ST> g_nonconstView = getNonconstLocalData(g);
  computeMaxValue(x, g_nonconstView[0]);

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());

  g_->assign(*g);
}

void SolutionMaxValueResponseFunction::
evaluateTangent(const double /* alpha */,
    const double /*beta*/,
    const double /*omega*/,
    const double current_time,
    bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  if (!gx.is_null() || !gp.is_null()) {
    evaluateGradient(current_time, x, xdot, xdotdot, p, deriv_p, g, gx, Teuchos::null, Teuchos::null, gp);
  }

  if (!gx.is_null() && !Vx.is_null()) {
    // compute gx = gx' * Vx.
    // Note: to avoid overwrite gx, we need to copy gx
    Teuchos::RCP<Thyra_MultiVector> gx_copy = gx->clone_mv();
    for (int i=0; i<Vx->domain()->dim(); ++i) {
      for (int j=0; j<Vx->domain()->dim(); ++j) {
        gx->col(i)->assign(gx_copy->col(i)->dot(*Vx->col(j)));
      }
    }
  }
}

void SolutionMaxValueResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    ParamVec* /*deriv_p*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  ST max_val;
  computeMaxValue(x, max_val);

  // Evaluate response g
  if (!g.is_null()) {
    Teuchos::ArrayRCP<ST> g_nonconstView = getNonconstLocalData(g);
    g_nonconstView[0] = max_val;
  }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    Teuchos::ArrayRCP<const ST> x_constView = getLocalData(x);
    Teuchos::ArrayRCP<ST> dg_dx_nonconstView = getNonconstLocalData(dg_dx->col(0));
    for (int i=0; i<x_constView.size(); ++i) {
      if (x_constView[i] == max_val) {
        dg_dx_nonconstView[i] = 1.0;
      } else {
        dg_dx_nonconstView[i] = 0.0;
      }
    }
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdotdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
void SolutionMaxValueResponseFunction::
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

void SolutionMaxValueResponseFunction::
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

void SolutionMaxValueResponseFunction::
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

void SolutionMaxValueResponseFunction::
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

void SolutionMaxValueResponseFunction::
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

void SolutionMaxValueResponseFunction::
computeMaxValue(const Teuchos::RCP<const Thyra_Vector>& x, ST& global_max)
{
  auto x_local = getLocalData(x);

  // Loop over nodes to find max value for equation eq
  int num_my_nodes = x_local.size() / neq;
  int index;
  ST my_max = std::numeric_limits<ST>::lowest();
  for (int node=0; node<num_my_nodes; node++) {
    if (interleavedOrdering == DiscType::Interleaved) {
      index = node*neq+eq;
    } else {
      index = node + eq*num_my_nodes;
    }
    if (x_local[index] > my_max) {
      my_max = x_local[index];
    }
  }

  // Check remainder (AGS: NOT SURE HOW THIS CODE GETS CALLED?)
  // LB: I believe this code would get called if equations at a given node are not
  //     forced to be on the same process, in which case neq may not divide the local
  //     dimension. I also believe Albany makes sure this does not happen, so I *think*
  //     these lines *should* be safe to remove...
  if (num_my_nodes*neq+eq < x_local.size()) {
    if (interleavedOrdering == DiscType::Interleaved) {
      index = num_my_nodes*neq+eq;
    } else {
      index = num_my_nodes + eq*num_my_nodes;
    }
    if (x_local[index] > my_max) {
      my_max = x_local[index];
    }
  }

  // Get max value across all proc's
  Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MAX, 1, &my_max, &global_max);
}

void
SolutionMaxValueResponseFunction::
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
