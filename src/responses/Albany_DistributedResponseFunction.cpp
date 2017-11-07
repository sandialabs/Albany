//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
