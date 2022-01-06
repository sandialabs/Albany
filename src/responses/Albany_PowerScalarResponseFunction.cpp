//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_PowerScalarResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Thyra_VectorStdOps.hpp"

using Teuchos::RCP;
using Teuchos::rcp;

Albany::PowerScalarResponseFunction::
PowerScalarResponseFunction(
  const Teuchos::RCP<const Teuchos_Comm>& commT,
  const Teuchos::RCP<ScalarResponseFunction>& response_,
  const double scalar_target_,
  const double exponent_) :
  SamplingBasedScalarResponseFunction(commT),
  response(response_),
  scalar_target(scalar_target_),
  exponent(exponent_)
{

}

void
Albany::PowerScalarResponseFunction::
setup()
{
  response->setup();
}

void
Albany::PowerScalarResponseFunction::
postRegSetup()
{
  response->postRegSetup();
}

Albany::PowerScalarResponseFunction::
~PowerScalarResponseFunction()
{
}

unsigned int
Albany::PowerScalarResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::PowerScalarResponseFunction::
evaluateResponse(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  g->assign(0);

  if (g_.is_null())
    g_ = Thyra::createMember(response->responseVectorSpace());

  if (f_.is_null())
    f_ = Thyra::createMember(response->responseVectorSpace());

  Teuchos::RCP<Thyra_Vector> g_i = Thyra::createMember(response->responseVectorSpace());
  response->evaluateResponse(current_time, x, xdot, xdotdot, p, g_i);

  f_->assign(*g_i);

  g->assign(pow(Thyra::get_ele(*g_i,0)-scalar_target, exponent));

  g_->assign(*g);
}

void
Albany::PowerScalarResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		Teuchos::Array<ParamVec>& p,
    const int parameter_index,
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

  // Create Thyra_Vectors for response function
  Teuchos::RCP<Thyra_Vector> g_i;
  Teuchos::RCP<Thyra_MultiVector> gx_i, gp_i; 

  auto vs_i = response->responseVectorSpace();

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
  response->evaluateTangent(alpha, beta, omega, current_time, sum_derivs,
        x, xdot, xdotdot, p, parameter_index, Vx, Vxdot, Vxdotdot, Vp, 
        g_i, gx_i, gp_i);

  double a;

  if (!g_i.is_null()) {
    a = exponent * pow(Thyra::get_ele(*g_i,0)-scalar_target, exponent-1);
  }
  else {
    a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);
  }

  // Copy results into combined result
  if (!g.is_null()) {
    g->assign(*g_i);
  }
  if (!gx.is_null()) {
    gx->update(a, *gx_i);
  }
  if (!gp.is_null()) {
    gp->update(a, *gp_i);
  }
}

void
Albany::PowerScalarResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
    const int parameter_index,
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

  auto vs_i = response->responseVectorSpace();

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
    dg_dp_i = Thyra::createMembers(vs_i, 1);
  }

  // Evaluate response function
  response->evaluateGradient(
          current_time, x, xdot, xdotdot, p, parameter_index, 
          g_i, dg_dx_i, dg_dxdot_i, dg_dxdotdot_i, dg_dp_i);

  double a;

  if (!g_i.is_null()) {
    a = exponent * pow(Thyra::get_ele(*g_i,0)-scalar_target, exponent-1);
  }
  else {
    a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);
  }

  // Copy results into combined result
  if (!g.is_null()) {
    g->assign(*g_i);
  }
  if (!dg_dx.is_null()) {
    dg_dx->update(a, *dg_dx_i);
  }
  if (!dg_dxdot.is_null()) {
    dg_dxdot->update(a, *dg_dxdot_i);
  }
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->update(a, *dg_dxdotdot_i);
  }
  if (!dg_dp.is_null()) {
    dg_dp->update(a, *dg_dp_i);
  }
}


void
Albany::PowerScalarResponseFunction::
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

  auto vs_i = response->responseVectorSpace();

  RCP<Thyra_MultiVector> dg_dp_i = Thyra::createMembers(dg_dp->range(), vs_i->dim());

  // Evaluate response function
  response->evaluateDistParamDeriv(
          current_time, x, xdot, xdotdot,
          param_array, dist_param_name,
          dg_dp_i);

  double a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);

  // Copy results into combined result
  dg_dp->update(a, *dg_dp_i);
}

void
Albany::PowerScalarResponseFunction::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                              "Error! evaluate_HessVecProd_xx is not yet implemented.\n");
  if (Hv_dp.is_null()) {
    return;
  }

  Hv_dp->assign(0.0);

  auto vs_i = response->responseVectorSpace();

  // Create Thyra_MultiVector for response derivative function
  RCP<Thyra_MultiVector> Hv_dp_i = Thyra::createMembers(Hv_dp->range(), vs_i->dim());

  // Evaluate response function
  response->evaluate_HessVecProd_xx(
          current_time, v, x, xdot, xdotdot,
          param_array,
          Hv_dp_i);
  
  double a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);
  double b = exponent * (exponent-1) * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-2);

  // Copy results into combined result
  Hv_dp->update(a, *Hv_dp_i);
}

void
Albany::PowerScalarResponseFunction::
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                              "Error! evaluate_HessVecProd_xp is not yet implemented.\n");
  if (Hv_dp.is_null()) {
    return;
  }

  Hv_dp->assign(0.0);

  auto vs_i = response->responseVectorSpace();

  // Create Thyra_MultiVector for response derivative function
  RCP<Thyra_MultiVector> Hv_dp_i = Thyra::createMembers(Hv_dp->range(), vs_i->dim());

  // Evaluate response function
  response->evaluate_HessVecProd_xp(
          current_time, v, x, xdot, xdotdot,
          param_array, dist_param_direction_name,
          Hv_dp_i);

  double a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);
  double b = exponent * (exponent-1) * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-2);

  // Copy results into combined result
  Hv_dp->update(a, *Hv_dp_i);
}

void
Albany::PowerScalarResponseFunction::
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                              "Error! evaluate_HessVecProd_px is not yet implemented.\n");    
  if (Hv_dp.is_null()) {
    return;
  }

  Hv_dp->assign(0.0);

  auto vs_i = response->responseVectorSpace();

  // Create Thyra_MultiVector for response derivative function
  RCP<Thyra_MultiVector> Hv_dp_i = Thyra::createMembers(Hv_dp->range(), vs_i->dim());

  // Evaluate response function
  response->evaluate_HessVecProd_px(
          current_time, v, x, xdot, xdotdot,
          param_array, dist_param_name,
          Hv_dp_i);

  double a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);
  double b = exponent * (exponent-1) * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-2);

  // Copy results into combined result
  Hv_dp->update(a, *Hv_dp_i);
}

void
Albany::PowerScalarResponseFunction::
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                              "Error! evaluate_HessVecProd_pp is not yet implemented.\n");  
  if (Hv_dp.is_null()) {
    return;
  }

  Hv_dp->assign(0.0);

  
  auto vs_i = response->responseVectorSpace();

  // Create Thyra_MultiVector for response derivative function
  RCP<Thyra_MultiVector> Hv_dp_i = Thyra::createMembers(Hv_dp->range(), vs_i->dim());

  // Evaluate response function
  response->evaluate_HessVecProd_pp(
          current_time, v, x, xdot, xdotdot,
          param_array, dist_param_name,
          dist_param_direction_name,
          Hv_dp_i);

  double a = exponent * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-1);
  double b = exponent * (exponent-1) * pow(Thyra::get_ele(*f_,0)-scalar_target, exponent-2);

  // Copy results into combined result
  Hv_dp->update(a, *Hv_dp_i);
}

void
Albany::PowerScalarResponseFunction::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> out)
{
  if (g_.is_null()) {
    *out << " the response has not been evaluated yet!";
    return;
  }

  std::size_t precision = 8;
  std::size_t value_width = precision + 4;
  
  *out << std::setw(value_width) << Thyra::get_ele(*g_,0) << "  = ( ";
  response->printResponse(out);
  *out << " - " << scalar_target << " ) ^ " << exponent;
}

void
Albany::PowerScalarResponseFunction::
updateTarget(double scalar_target_)
{
  scalar_target = scalar_target_;
}

void
Albany::PowerScalarResponseFunction::
updateExponent(double exponent_)
{
  exponent = exponent_;
}