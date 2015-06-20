//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DistributedResponseFunction.hpp"

//IK, 9/13/14: Epetra ifdef'ed out except if SG and MP if ALBANY_EPETRA_EXE set to off.

#if defined(ALBANY_EPETRA)
void
Albany::DistributedResponseFunction::
evaluateDerivative(
  const double current_time,
  const Epetra_Vector* xdot,
  const Epetra_Vector* xdotdot,
  const Epetra_Vector& x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Epetra_Vector* g,
  const EpetraExt::ModelEvaluator::Derivative& dg_dx,
  const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
  const EpetraExt::ModelEvaluator::Derivative& dg_dxdotdot,
  const EpetraExt::ModelEvaluator::Derivative& dg_dp)
{
  this->evaluateGradient(
    current_time, xdot, xdotdot, x, p, deriv_p, g,
    dg_dx.getLinearOp().get(), dg_dxdot.getLinearOp().get(),
    dg_dxdotdot.getLinearOp().get(), dg_dp.getMultiVector().get());
}
#endif

void
Albany::DistributedResponseFunction::
evaluateDerivativeT(
  const double current_time,
  const Tpetra_Vector* xdotT,
  const Tpetra_Vector* xdotdotT,
  const Tpetra_Vector& xT,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Tpetra_Vector* gT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdotT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dpT)
{
  Tpetra_Operator* dg_dxp;
  if(dg_dxT.isEmpty()){
    dg_dxp = NULL;
  }
  else {
    Teuchos::RCP<Tpetra_Operator> dgdxT = ConverterT::getTpetraOperator(dg_dxT.getLinearOp());
    dg_dxp = dgdxT.get();
  }

  Tpetra_Operator* dg_dxdotp;
  if(dg_dxdotT.isEmpty()){
    dg_dxdotp = NULL;
  }
  else {
    Teuchos::RCP<Tpetra_Operator> dgdxdotT = ConverterT::getTpetraOperator(dg_dxdotT.getLinearOp());
    dg_dxdotp = dgdxdotT.get();
  }

  Tpetra_Operator* dg_dxdotdotp;
  if(dg_dxdotdotT.isEmpty()){
    dg_dxdotdotp = NULL;
  }
  else {
    Teuchos::RCP<Tpetra_Operator> dgdxdotdotT = ConverterT::getTpetraOperator(dg_dxdotdotT.getLinearOp());
    dg_dxdotdotp = dgdxdotdotT.get();
  }

  Tpetra_MultiVector* dg_dpp;
  if(dg_dpT.isEmpty()){
    dg_dpp = NULL;
  }
  else {
    Teuchos::RCP<Tpetra_MultiVector> dgdpT = ConverterT::getTpetraMultiVector(dg_dpT.getMultiVector());
    dg_dpp = dgdpT.get();
  }

  this->evaluateGradientT(
    current_time, xdotT, xdotdotT, xT, p, deriv_p, gT,
    dg_dxp, dg_dxdotp, dg_dxdotdotp, dg_dpp);
}

#ifdef ALBANY_SG
void
Albany::DistributedResponseFunction::
evaluateSGDerivative(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdotdot,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp)
{
  this->evaluateSGGradient(
    current_time, sg_xdot, sg_xdotdot, sg_x, p, sg_p_index, sg_p_vals, deriv_p,
    sg_g, sg_dg_dx.getLinearOp().get(), sg_dg_dxdot.getLinearOp().get(),
    sg_dg_dxdotdot.getLinearOp().get(), sg_dg_dp.getMultiVector().get());
}
#endif 
#ifdef ALBANY_ENSEMBLE 

void
Albany::DistributedResponseFunction::
evaluateMPDerivative(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdotdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp)
{
  this->evaluateMPGradient(
    current_time, mp_xdot, mp_xdotdot, mp_x, p, mp_p_index, mp_p_vals, deriv_p,
    mp_g, mp_dg_dx.getLinearOp().get(), mp_dg_dxdot.getLinearOp().get(),
    mp_dg_dxdotdot.getLinearOp().get(), mp_dg_dp.getMultiVector().get());
}
#endif
