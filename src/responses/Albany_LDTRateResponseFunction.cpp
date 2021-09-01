//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_LDTRateResponseFunction.hpp"

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

#include "Teuchos_SerialDenseHelpers.hpp"
#include "Teuchos_SerialDenseSolver.hpp"
#include <Teuchos_TwoDArray.hpp>

using Teuchos::ScalarTraits;
using Teuchos::SerialDenseMatrix;
using Teuchos::SerialDenseVector;

namespace Albany
{

LDTRateResponseFunction::
LDTRateResponseFunction(const Teuchos::RCP<const Application>& app,
                               Teuchos::ParameterList& responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app)
{
  n_parameters = responseParams.get<int>("Number Of Parameters", 1);
  theta_0 = Teuchos::rcp( new Teuchos::SerialDenseVector<int, double>(n_parameters) );
  C = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int, double>(n_parameters,n_parameters) );
  invC = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int, double>(n_parameters,n_parameters) );

  Teuchos::TwoDArray<double> C_data(n_parameters, n_parameters, 0);
  if (responseParams.isParameter("Covariance Matrix")) {
    C_data = responseParams.get<Teuchos::TwoDArray<double>>("Covariance Matrix");
  }
  else {
    for (int i=0; i<n_parameters; i++)
      C_data(i,i) = 1.;
  }

  Teuchos::Array<double> theta_0_data;
  if (responseParams.isParameter("Mean")) {
    theta_0_data = responseParams.get<Teuchos::Array<double> >("Mean");
  }
  else {
    for (int i=0; i<n_parameters; i++)
      theta_0_data.push_back(0.);
  }

  for (int i=0; i<n_parameters; i++)
    for (int j=0; j<n_parameters; j++)
      (*C)(i,j) = C_data(i,j);

  for (int i=0; i<n_parameters; i++)
    (*theta_0)(i) = theta_0_data[i];
  
  // Fill invC:
  SerialDenseVector<int, double> tmp1(n_parameters), tmp2(n_parameters);
  Teuchos::SerialDenseSolver<int, double> solver;
  solver.setMatrix( C );
  for (int i=0; i<n_parameters; i++) {
    tmp1.putScalar( 0.0 );
    tmp1(i) = 1.0;
    solver.setVectors( Teuchos::rcp( &tmp2, false ), Teuchos::rcp( &tmp1, false ) );
    solver.factor();
    solver.solve();
    Teuchos::setCol<int, double>( tmp2, i, *invC );
  }
}

void LDTRateResponseFunction::setup()
{
}

unsigned int LDTRateResponseFunction::
numResponses() const
{
  return 1;
}

void LDTRateResponseFunction::
evaluateResponseImpl (
    const Teuchos::Array<ParamVec> &p,
		Thyra_Vector& g)
{
  // I(\theta) = 1/2 \|\theta-\theta_0 \|^2_{C^{-1}}
  //           = 1/2 (\theta-\theta_0)^T C^{-1} (\theta-\theta_0)

  typedef SerialDenseVector<int, double> DVector;

  DVector theta(n_parameters), tmp1(n_parameters), tmp2(n_parameters);
/*
  ParamVec params_l = p[0];
  unsigned int num_cols_p_l = params_l.size();

  TEUCHOS_TEST_FOR_EXCEPTION(
      num_cols_p_l != n_parameters,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::LDTRateResponseFunction::evaluateResponse():  "
          << "The number of parameter in the parameter vector "
          << num_cols_p_l
          << " is not consistent with the number of parameter of the xml file "
          << n_parameters
          << std::endl);  
*/
  for (int i=0; i<n_parameters; i++) {
    theta(i) = p[i][0].family->getValue<PHAL::AlbanyTraits::Residual>();
  }

  for (int i=0; i<n_parameters; i++) {
    tmp1(i) = theta(i) - (*theta_0)(i);
  }

  tmp2.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *invC, tmp1, 0.0 );

  double I = 0.5 * tmp2.dot(tmp1);

  g.assign(I);
}

void LDTRateResponseFunction::
evaluateGradientImpl (
    const Teuchos::Array<ParamVec> &p,
		SerialDenseVector<int, double>& dgdp)
{
  // I(\theta) = 1/2 \|\theta-\theta_0 \|^2_{C^{-1}}
  //           = 1/2 (\theta-\theta_0)^T C^{-1} (\theta-\theta_0)

  typedef SerialDenseVector<int, double> DVector;

  DVector theta(n_parameters), tmp1(n_parameters);
/*
  ParamVec params_l = p[0];
  unsigned int num_cols_p_l = params_l.size();

  TEUCHOS_TEST_FOR_EXCEPTION(
      num_cols_p_l != n_parameters,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Albany::LDTRateResponseFunction::evaluateGradientImpl():  "
          << "The number of parameter in the parameter vector "
          << num_cols_p_l
          << " is not consistent with the number of parameter of the xml file "
          << n_parameters
          << std::endl);  
*/
  for (int i=0; i<n_parameters; i++) {
    theta(i) = p[i][0].family->getValue<PHAL::AlbanyTraits::Residual>();
  }

  for (int i=0; i<n_parameters; i++) {
    tmp1(i) = theta(i) - (*theta_0)(i);
  }

  dgdp.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *invC, tmp1, 0.0 );
}

void LDTRateResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  // Evaluate response g
  if (!g.is_null()) {
    evaluateResponseImpl(p,*g);
  }

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());

  g_->assign(*g);
}

void LDTRateResponseFunction::
evaluateTangent(const double /*alpha*/,
		const double beta,
		const double /* omega */,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& p,
    int parameter_index,
		ParamVec* deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vx*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  // Evaluate response g
  if (!g.is_null()) {
    evaluateResponseImpl(p,*g);
  }
  
  if (!gx.is_null()) {
    gx->assign(0.0);
  }

  if (!gp.is_null()) {
    // dgdp = dg/dp
    SerialDenseVector<int, double> dgdp(n_parameters);
    evaluateGradientImpl (p, dgdp);

    // gp = (dg/dp)^T*Vp
    /*
    if (Teuchos::nonnull(Vp)) {
      Teuchos::ArrayRCP<const double> Vp_constView;
      for (int k = 0; k < Vp->domain()->dim(); k++) {
        Vp_constView                     = getLocalData(Vp->col(k));
        double tmp_gp = 0;
        for (int i=0; i<n_parameters; i++)
          tmp_gp += Vp_constView[i]*tmp2(i);
        gp->col(k)->assign(tmp_gp);
      }
    }
    */
   /*
    if (Teuchos::nonnull(Vp)) {
      Teuchos::ArrayRCP<const double> Vp_constView;
      for (int k = 0; k < Vp->domain()->dim(); k++) {
        Vp_constView                     = getLocalData(Vp->col(k));
        double tmp_gp = 0;
        for (int i=0; i<n_parameters; i++)
          tmp_gp += Vp_constView[i]*dgdp(i);
        gp->col(k)->assign(tmp_gp);
      }
    }
  */
    Thyra::set_ele(0, dgdp(parameter_index), gp->col(0).ptr());
  
  }
}

//! Evaluate distributed parameter derivative dg/dp
void LDTRateResponseFunction::
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

void LDTRateResponseFunction::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void LDTRateResponseFunction::
evaluate_HessVecProd_xp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_direction_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void LDTRateResponseFunction::
evaluate_HessVecProd_px(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void LDTRateResponseFunction::
evaluate_HessVecProd_pp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& /*dist_param_name*/,
    const std::string& /*dist_param_direction_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    typedef SerialDenseVector<int, double> DVector;

    DVector tmp1(n_parameters), tmp2(n_parameters);

    for (int i=0; i<n_parameters; i++) {
      tmp1(i) = Thyra::get_ele(*v->col(0),i);
    }

    tmp2.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *invC, tmp1, 0.0 );

    for (int i=0; i<n_parameters; i++) {
      Thyra::set_ele(i, tmp2(i), Hv_dp->col(0).ptr());
    }
    /*
    for (int k = 0; k < v->domain()->dim(); k++) {
      for (int i=0; i<n_parameters; i++) {
        tmp1(i) = Thyra::get_ele(*v->col(k),i);
      }

      tmp2.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *invC, tmp1, 0.0 );

      for (int i=0; i<n_parameters; i++) {
        Thyra::set_ele(i, tmp2(i), Hv_dp->col(k).ptr());
      }
    }
    */
  }
}

void LDTRateResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* /*deriv_p*/,
		const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  // Evaluate response g
  if (!g.is_null()) {
    evaluateResponseImpl(p,*g);
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
    // dgdp = dg/dp
    SerialDenseVector<int, double> dgdp(n_parameters);
    evaluateGradientImpl (p, dgdp);

    for (int i=0; i<n_parameters; i++) {
      dg_dp->col(i)->assign(dgdp(i));
    }
  }
}

void
LDTRateResponseFunction::
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
