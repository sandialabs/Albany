//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_CumulativeScalarResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_ThyraUtils.hpp"

using Teuchos::RCP;
using Teuchos::rcp;

Albany::CumulativeScalarResponseFunction::
CumulativeScalarResponseFunction(
  const Teuchos::RCP<const Teuchos_Comm>& commT,
  const Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> >& responses_) :
  SamplingBasedScalarResponseFunction(commT),
  responses(responses_),
  num_responses(0)
{
  if(responses.size() > 0) {
    num_responses = responses[0]->numResponses();

    // Check that all responses have the same vector space 
    auto vs = responses[0]->responseVectorSpace(); 
    for (int iresp=1; iresp<num_responses; ++iresp) {
      TEUCHOS_TEST_FOR_EXCEPTION(!responses[iresp]->responseVectorSpace()->isCompatible(*vs), std::runtime_error,
                                 "Error! All responses in CumulativeScalarResponseFunction must have compatible vector spaces.\n");
    }
  }
}

void
Albany::CumulativeScalarResponseFunction::
setup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction> > ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->setup();
  }
}

void
Albany::CumulativeScalarResponseFunction::
postRegSetup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction> > ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->postRegSetup();
  }
}

Albany::CumulativeScalarResponseFunction::
~CumulativeScalarResponseFunction()
{
}

unsigned int
Albany::CumulativeScalarResponseFunction::
numResponses() const 
{
  return num_responses;
}

void
Albany::CumulativeScalarResponseFunction::
evaluateResponse(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  g->assign(0);

  for (unsigned int i=0; i<responses.size(); i++) {
    // Create Thyra_Vector for response function
    Teuchos::RCP<Thyra_Vector> g_i = Thyra::createMember(responses[i]->responseVectorSpace());
  
    // Evaluate response function
    responses[i]->evaluateResponse(current_time, x, xdot, xdotdot, p, g_i);

    // Add result into cumulative result
    g->update(1.0,*g_i);
  }
}

void
Albany::CumulativeScalarResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
		const Teuchos::RCP<Thyra_Vector>& g,
		const Teuchos::RCP<Thyra_MultiVector>& gx,
		const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  //zero-out vecotres
  if (!g.is_null()) {
    g->assign(0);
  }
  if (!gx.is_null()) {
    gx->assign(0);
  }
  if (!gp.is_null()) {
    gp->assign(0);
  }

  for (unsigned int i=0; i<responses.size(); i++) {
    // Create Thyra_Vectors for response function
    Teuchos::RCP<Thyra_Vector> g_i;
    Teuchos::RCP<Thyra_MultiVector> gx_i, gp_i; 

    // Note: all vector spaces should be the same, so you could just
    // always use response[0]->responseVectorSpace()
    auto vs_i = responses[i]->responseVectorSpace();

    if (!g.is_null()) {
      g_i = Thyra::createMember(vs_i);
    }
    if (!gx.is_null()) {
      gx_i = Thyra::createMembers(vs_i,gx->domain()->dim());
    }
    if (!gp.is_null()) {
      gp_i = Thyra::createMembers(vs_i,gp->domain()->dim());
    }

    // Evaluate response function
    responses[i]->evaluateTangent(alpha, beta, omega, current_time, sum_derivs,
				  x, xdot, xdotdot, p, deriv_p, Vx, Vxdot, Vxdotdot, Vp, 
          g_i, gx_i, gp_i);

    // Copy results into combined result
    if (!g.is_null()) {
      g->update(1.0, *g_i);
    }
    if (!gx.is_null()) {
      gx->update(1.0, *gx_i);
    }
    if (!gp.is_null()) {
      gp->update(1.0, *gp_i);
    }
  }
}

void
Albany::CumulativeScalarResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Teuchos::RCP<Thyra_Vector>& g,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!g.is_null()) {
    g->assign(0.0);
  }
  if (!dg_dx.is_null()) {
    dg_dx->assign(0.0);
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

  for (unsigned int i=0; i<responses.size(); i++) {
    // Note: all vector spaces should be the same, so you could just
    // always use response[0]->responseVectorSpace()
    auto vs_i = responses[i]->responseVectorSpace();

    // Create Thyra_Vectors for response function
    RCP<Thyra_Vector> g_i;
    RCP<Thyra_MultiVector> dg_dx_i, dg_dxdot_i, dg_dxdotdot_i, dg_dp_i;
    if (!g.is_null()) {
      g_i = Thyra::createMember(vs_i);
    }
    if (!dg_dx.is_null()) {
      dg_dx_i = Thyra::createMembers(dg_dx->range(), vs_i->dim());
    }
    if (!dg_dxdot.is_null()) {
      dg_dxdot_i = Thyra::createMembers(dg_dxdot->range(), vs_i->dim());
    }
    if (!dg_dxdotdot.is_null()) {
      dg_dxdotdot_i = Thyra::createMembers(dg_dxdot->range(), vs_i->dim());
    }
    if (!dg_dp.is_null()) {
      dg_dp_i = Thyra::createMembers(vs_i, num_responses);
    }

    // Evaluate response function
    responses[i]->evaluateGradient(
            current_time, x, xdot, xdotdot, p, deriv_p, 
            g_i, dg_dx_i, dg_dxdot_i, dg_dxdotdot_i, dg_dp_i);

    // Copy results into combined result
    if (!g.is_null()) {
      g->update(1.0, *g_i);
    }
    if (!dg_dx.is_null()) {
      dg_dx->update(1.0, *dg_dx_i);
    }
    if (!dg_dxdot.is_null()) {
      dg_dxdot->update(1.0, *dg_dxdot_i);
    }
    if (!dg_dxdotdot.is_null()) {
      dg_dxdotdot->update(1.0, *dg_dxdotdot_i);
    }
    if (!dg_dp.is_null()) {
      dg_dp->update(1.0, *dg_dp_i);
    }
  }
}


void
Albany::CumulativeScalarResponseFunction::
evaluateDistParamDeriv(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (dg_dp.is_null()) {
    return;
  }

  dg_dp->assign(0.0);

  for (unsigned int i=0; i<responses.size(); i++) {
    auto vs_i = responses[i]->responseVectorSpace();

    // Create Thyra_MultiVector for response derivative function
    RCP<Thyra_MultiVector> dg_dp_i = Thyra::createMembers(dg_dp->range(), vs_i->dim());

    // Evaluate response function
    responses[i]->evaluateDistParamDeriv(
           current_time, x, xdot, xdotdot,
           param_array, dist_param_name,
           dg_dp_i);

    // Copy results into combined result
    dg_dp->update(1.0, *dg_dp_i);
  }
}
