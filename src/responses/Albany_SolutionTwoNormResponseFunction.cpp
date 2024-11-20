//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionTwoNormResponseFunction.hpp"

#include "Thyra_VectorStdOps.hpp"

Albany::SolutionTwoNormResponseFunction::
SolutionTwoNormResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT) :
  SamplingBasedScalarResponseFunction(commT)
{
}

Albany::SolutionTwoNormResponseFunction::
~SolutionTwoNormResponseFunction()
{
}

unsigned int
Albany::SolutionTwoNormResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::SolutionTwoNormResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const Teuchos::RCP<Thyra_Vector>& g)
{
  Teuchos::ScalarTraits<ST>::magnitudeType twonorm = x->norm_2();
  g->assign(twonorm);

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());

  g_->assign(*g);
}

void
Albany::SolutionTwoNormResponseFunction::
evaluateTangent(const double alpha, 
		const double /*beta*/,
		const double /*omega*/,
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
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  if (!g.is_null()) {
    g->assign(nrm);
  }

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  // dg/dx = 1/||x|| * x^T
  if (!gx.is_null()) {
    if (!Vx.is_null()) {
      // compute gx = x' * Vx. x is a vector, Vx a multivector,
      // so gx is a MV with range->dim()=1, each column being
      // the dot product x->dot(*Vx->col(j))
      for (int j=0; j<Vx->domain()->dim(); ++j) {
        gx->col(j)->assign(x->dot(*Vx->col(j)));
      }
    } else {
      // V_StV stands for V_out = Scalar * V_in
      Thyra::V_StV(gx->col(0).ptr(),alpha/nrm,*x);
    }
  }

  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

void
Albany::SolutionTwoNormResponseFunction::
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
  Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();

  // Evaluate response g
  if (!g.is_null()) {
    g->assign(nrm);
  }
  
  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    // V_StV stands for V_out = Scalar * V_in
    Thyra::V_StV(dg_dx->col(0).ptr(),1.0/nrm,*x);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdot
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
Albany::SolutionTwoNormResponseFunction::
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
Albany::SolutionTwoNormResponseFunction::
evaluate_HessVecProd_xx(
    const double /* current_time */,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /* xdot */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdotdot */,
    const Teuchos::Array<ParamVec>& /* param_array */,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dxdx)
{
  if (!Hv_dxdx.is_null()) {
    Teuchos::ScalarTraits<ST>::magnitudeType nrm = x->norm_2();
    TEUCHOS_TEST_FOR_EXCEPTION (nrm == 0,
          std::runtime_error,
          "Second derivative not defined when x is zero");

    Teuchos::ScalarTraits<ST>::magnitudeType nrm3 = std::pow(nrm,3);
    for (int j=0; j<Hv_dxdx->domain()->dim(); ++j) {
      // Evaluate Hv_dxdx_j = v_j/||x|| - (v_j,x) x/||x||^3 where (.,.) is the usual dot product:
      Thyra::V_StVpStV(Hv_dxdx->col(j).ptr(),1.0/nrm,*v->col(j),-1.0*x->dot(*v->col(j))/nrm3,*x);
    }
  }
}

void
Albany::SolutionTwoNormResponseFunction::
evaluate_HessVecProd_xp(
    const double /* current_time */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* v */,
    const Teuchos::RCP<const Thyra_Vector>& /* x */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdot */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdotdot */,
    const Teuchos::Array<ParamVec>& /* param_array */,
    const std::string& /* dist_param_direction_name */,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::SolutionTwoNormResponseFunction::
evaluate_HessVecProd_px(
    const double /* current_time */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* v */,
    const Teuchos::RCP<const Thyra_Vector>& /* x */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdot */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdotdot */,
    const Teuchos::Array<ParamVec>& /* param_array */,
    const std::string& /* dist_param_name */,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::SolutionTwoNormResponseFunction::
evaluate_HessVecProd_pp(
    const double /* current_time */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* v */,
    const Teuchos::RCP<const Thyra_Vector>& /* x */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdot */,
    const Teuchos::RCP<const Thyra_Vector>& /* xdotdot */,
    const Teuchos::Array<ParamVec>& /* param_array */,
    const std::string& /* dist_param_name */,
    const std::string& /* dist_param_direction_name */,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::SolutionTwoNormResponseFunction::
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
