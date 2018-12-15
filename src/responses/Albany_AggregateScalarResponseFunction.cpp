//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_AggregateScalarResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Thyra_VectorBase.hpp"

#include "Thyra_DefaultProductVectorSpace.hpp"

Albany::AggregateScalarResponseFunction::
AggregateScalarResponseFunction(
  const Teuchos::RCP<const Teuchos_Comm>& commT,
  const Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> >& responses_) :
  SamplingBasedScalarResponseFunction(commT),
  responses(responses_)
{
}

void
Albany::AggregateScalarResponseFunction::
setup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction> > ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->setup();
  }

  // Now that all responses are setup, build the product vector space
  Teuchos::Array<Teuchos::RCP<const Thyra_VectorSpace>> vss(responses.size());
  for (int i=0; i<responses.size(); ++i) {
    vss[i] = responses[i]->responseVectorSpace();
  }

  productVectorSpace = Thyra::productVectorSpace(vss());
}

void
Albany::AggregateScalarResponseFunction::
postRegSetup()
{
  typedef Teuchos::Array<Teuchos::RCP<ScalarResponseFunction> > ResponseArray;
  for (ResponseArray::iterator it = responses.begin(), it_end = responses.end(); it != it_end; ++it) {
    (*it)->postRegSetup();
  }
}

Albany::AggregateScalarResponseFunction::
~AggregateScalarResponseFunction()
{
}

unsigned int
Albany::AggregateScalarResponseFunction::
numResponses() const 
{
  unsigned int n = 0;
  for (int i=0; i<responses.size(); i++)
    n += responses[i]->numResponses();
  return n;
}

void
Albany::AggregateScalarResponseFunction::
evaluateResponse(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  if (g.is_null()) {
    return;
  }

  // NOTE: You CANNOT use ProduceVector's if you want to maintain support for EpetraExt::ModelEvaluator,
  //       since that class has no knowledge of vector spaces, and would try to build a monolithic map
  //       for the aggregate response. For now, stick with monolithic responses and manual copies

  /*
   * // Cast response to product vector
   * auto g_prod = getProductVector(g);
   * 
   * for (unsigned int i=0; i<responses.size(); i++) {
   *   // Evaluate response function
   *   responses[i]->evaluateResponse(current_time, x, xdot, xdotdot, p, g_prod->getNonconstVectorBlock(i));
   * }
   */

  Teuchos::ArrayRCP<ST> g_data = getNonconstLocalData(g); // We already checked g is not null
  Teuchos::ArrayRCP<const ST> gi_data;

  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {
    // Create Thyra_Vector for response function
    Teuchos::RCP<Thyra_Vector> g_i = Thyra::createMember(productVectorSpace->getBlock(i));
    g_i->assign(0.0); 
    
    gi_data = getLocalData(g_i.getConst());

     // Evaluate response function
    responses[i]->evaluateResponse(current_time, x, xdot, xdotdot, p, g_i);

    // Copy into the monolithic vector
    for (unsigned int j=0; j<responses[i]->numResponses(); ++j) {
      g_data[offset+j] = gi_data[j];
    }

    // Update offset
    offset += responses[i]->numResponses();
  }
}

void
Albany::AggregateScalarResponseFunction::
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
  // NOTE: You CANNOT use ProduceVector's if you want to maintain support for EpetraExt::ModelEvaluator,
  //       since that class has no knowledge of vector spaces, and would try to build a monolithic map
  //       for the aggregate response. For now, stick with monolithic responses and manual copies

/*
 *   // Cast response (and derivs) to product (multi)vector
 *   auto g_prod = getProductVector(g);
 *   auto gx_prod = getProductMultiVector(gx);
 *   auto gp_prod = getProductMultiVector(gp);
 * 
 *   for (unsigned int i=0; i<responses.size(); i++) {
 *     // Create Thyra_(Multi)Vector's for response function
 *     Teuchos::RCP<Thyra_Vector> g_i;
 *     Teuchos::RCP<Thyra_MultiVector> gx_i, gp_i;
 *     if (!g.is_null()) {
 *       g_i = g_prod->getNonconstVectorBlock(i);
 *     }
 *     if (!gx.is_null()) {
 *       gx_i = gx_prod->getNonconstMultiVectorBlock(i);
 *     }
 *     if (!gp.is_null()) {
 *       gp_i = gp_prod->getNonconstMultiVectorBlock(i);
 *     }
 * 
 *     // Evaluate response function
 *     responses[i]->evaluateTangent(alpha, beta, omega, current_time, sum_derivs,
 *           x, xdot, xdotdot, p, deriv_p, Vx, Vxdot, Vxdotdot, Vp, 
 *           g_i, gx_i, gp_i);
 *   }
 */

  Teuchos::ArrayRCP<ST> g_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> gx_data, gp_data;

  if (!g.is_null())  { g_data  = getNonconstLocalData(g);  }
  if (!gx.is_null()) { gx_data = getNonconstLocalData(gx); }
  if (!gp.is_null()) { gp_data = getNonconstLocalData(gp); }

  Teuchos::ArrayRCP<const ST> gi_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> gxi_data, gpi_data;

  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {
    auto vs_i = productVectorSpace->getBlock(i);

    // Create Thyra_(Multi)Vector's for response function
    Teuchos::RCP<Thyra_Vector> g_i;
    Teuchos::RCP<Thyra_MultiVector> gx_i, gp_i;

    if (!g.is_null()) {
      g_i = Thyra::createMember(vs_i);
      gi_data = getLocalData(g_i.getConst());
    }
    if (!gx.is_null()) {
      gx_i = Thyra::createMembers(vs_i,gx->domain()->dim());
      gxi_data = getLocalData(gx_i.getConst());
    }
    if (!gp.is_null()) {
      gp_i = Thyra::createMembers(vs_i,gp->domain()->dim());
      gpi_data = getLocalData(gp_i.getConst());
    }
 
    // Evaluate response function
    responses[i]->evaluateTangent(alpha, beta, omega, current_time, sum_derivs,
          x, xdot, xdotdot, p, deriv_p, Vx, Vxdot, Vxdotdot, Vp, 
          g_i, gx_i, gp_i);

    // Copy into the monolithic (multi)vectors
    for (unsigned int j=0; j<responses[i]->numResponses(); ++j) {
      if (!g.is_null()) {
        g_data[offset+j] = gi_data[j];
      }
      if (!gx.is_null()) {
        for (int col=0; col<gx->domain()->dim(); ++col) {
          gx_data[col][offset+j] = gxi_data[col][j];
        }
      }
      if (!gp.is_null()) {
        for (int col=0; col<gp->domain()->dim(); ++col) {
          gp_data[col][offset+j] = gpi_data[col][j];
        }
      }
    }

    // Update the offset
    offset += vs_i->dim();
  }
}

void
Albany::AggregateScalarResponseFunction::
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
  // NOTE: You CANNOT use ProduceVector's if you want to maintain support for EpetraExt::ModelEvaluator,
  //       since that class has no knowledge of vector spaces, and would try to build a monolithic map
  //       for the aggregate response. For now, stick with monolithic responses and manual copies

  /*
   * // Cast response (and param deriv) to product (multi)vector
   * auto g_prod     = getProductVector(g);
   */

  Teuchos::ArrayRCP<ST> g_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> dgdp_data;

  if (!g.is_null())     { g_data    = getNonconstLocalData(g);     }
  if (!dg_dp.is_null()) { dgdp_data = getNonconstLocalData(dg_dp); }

  Teuchos::ArrayRCP<const ST> g_i_data;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> dgdp_i_data;

  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {
    auto vs_i = productVectorSpace->getBlock(i);

    // dg_dx, dg_dxdot, and dg_dxdotdot for the i-th response are simply
    // a subview of the columns of the corresponding input MV's, at the proper offset
    Teuchos::Range1D colRange(offset, offset+vs_i->dim()-1);

    // Create Thyra_(Multi)Vector's for response function
    Teuchos::RCP<Thyra_Vector> g_i;
    Teuchos::RCP<Thyra_MultiVector> dgdx_i, dgdxdot_i, dgdxdotdot_i, dgdp_i;
    if (!g.is_null()) {
      g_i = Thyra::createMember(vs_i);
      g_i_data = getLocalData(g_i.getConst());
    }
    if (!dg_dx.is_null()) {
      dgdx_i = dg_dx->subView(colRange);
    }
    if (!dg_dxdot.is_null()) {
      dgdxdot_i = dg_dxdot->subView(colRange);
    }
    if (!dg_dxdotdot.is_null()) {
      dgdxdotdot_i = dg_dxdotdot->subView(colRange);
    }
    if (!dg_dp.is_null()) {
      dgdp_i = Thyra::createMembers(vs_i,dg_dp->domain()->dim());
      dgdp_i_data = getLocalData(dgdp_i.getConst());
    }

    // Evaluate response function
    responses[i]->evaluateGradient(current_time, x, xdot, xdotdot, p, deriv_p, 
                                   g_i, dgdx_i, dgdxdot_i, dgdxdotdot_i, dgdp_i);

    // Copy into the monolithic (multi)vectors
    for (unsigned int j=0; j<responses[i]->numResponses(); ++j) {
      if (!g.is_null()) {
        g_data[offset+j] = g_i_data[j];
      }
      if (!dg_dp.is_null()) {
        for (int col=0; col<dg_dp->domain()->dim(); ++col) {
          dgdp_data[col][offset+j] = dgdp_i_data[col][j];
        }
      }
    }

    // Update the offset
    offset += vs_i->dim();
  }
}

void
Albany::AggregateScalarResponseFunction::
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

  unsigned int offset = 0;
  for (unsigned int i=0; i<responses.size(); i++) {
    auto vs_i = productVectorSpace->getBlock(i);

    // dg_dp for the i-th response is simply a subview of the columns
    // of the corresponding input MV, at the proper offset
    Teuchos::Range1D colRange(offset, offset+vs_i->dim()-1);

    // Create Thyra_MultiVector for response function
    Teuchos::RCP<Thyra_MultiVector> dg_dp_i;
    if (!dg_dp.is_null()) {
      dg_dp_i = dg_dp->subView(colRange);
    }

    // Evaluate response derivative
    responses[i]->evaluateDistParamDeriv(
            current_time, x, xdot, xdotdot,
            param_array, dist_param_name,
            dg_dp_i);

    // Update the offset
    offset += vs_i->dim();
  }
}
