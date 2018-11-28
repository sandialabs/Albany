//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_DistObject.hpp"
#include "Thyra_SpmdVectorBase.hpp"

#include <limits>

Albany::SolutionMaxValueResponseFunction::
SolutionMaxValueResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT,
				 int neq_, int eq_, bool interleavedOrdering_) :
  SamplingBasedScalarResponseFunction(commT),
  commT_(commT), 
  neq(neq_), eq(eq_), interleavedOrdering(interleavedOrdering_)
{
}

Albany::SolutionMaxValueResponseFunction::
~SolutionMaxValueResponseFunction()
{
}

unsigned int
Albany::SolutionMaxValueResponseFunction::
numResponses() const 
{
  return 1;
}


void
Albany::SolutionMaxValueResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  Teuchos::ArrayRCP<ST> g_nonconstView = Albany::getNonconstLocalData(g);
  computeMaxValue(x, g_nonconstView[0]);
}

void
Albany::SolutionMaxValueResponseFunction::
evaluateTangent(const double alpha, 
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

void
Albany::SolutionMaxValueResponseFunction::
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
    Teuchos::ArrayRCP<ST> g_nonconstView = Albany::getNonconstLocalData(g);
    g_nonconstView[0] = max_val;
  }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
    Teuchos::ArrayRCP<ST> dg_dx_nonconstView = Albany::getNonconstLocalData(dg_dx->col(0));
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
void
Albany::SolutionMaxValueResponseFunction::
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

void
Albany::SolutionMaxValueResponseFunction::
computeMaxValue(const Teuchos::RCP<const Thyra_Vector>& x, ST& global_max)
{
  auto x_local = Albany::getLocalData(x);
  
  // Loop over nodes to find max value for equation eq
  int num_my_nodes = x_local.size() / neq;
  int index;
  ST my_max = std::numeric_limits<ST>::lowest();
  for (int node=0; node<num_my_nodes; node++) {
    if (interleavedOrdering) {
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
    if (interleavedOrdering) {
      index = num_my_nodes*neq+eq;
    } else {
      index = num_my_nodes + eq*num_my_nodes;
    }
    if (x_local[index] > my_max) {
      my_max = x_local[index];
    }
  }

  // Get max value across all proc's
  Teuchos::reduceAll(*commT_, Teuchos::REDUCE_MAX, 1, &my_max, &global_max); 
}
