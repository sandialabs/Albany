//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_ScalarResponseFunction.hpp"
#include "Teuchos_TestForException.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_LocalMap.h"
#endif

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map>
Albany::ScalarResponseFunction::
responseMap() const
{
  int num_responses = this->numResponses();
  Teuchos::RCP<Epetra_Comm> comm = Albany::createEpetraCommFromTeuchosComm(commT);
  Teuchos::RCP<const Epetra_LocalMap> response_map =
    Teuchos::rcp(new Epetra_LocalMap(num_responses, 0, *comm));
  return response_map;
}
#endif

//Tpetra version of above
Teuchos::RCP<const Tpetra_Map>
Albany::ScalarResponseFunction::
responseMapT() const
{
  int num_responses = this->numResponses();
  //the following is needed to create Tpetra local map...
  Tpetra::LocalGlobal lg = Tpetra::LocallyReplicated;
  Teuchos::RCP<const Tpetra_Map> response_mapT =
    Teuchos::rcp(new Tpetra_Map(num_responses, 0, commT, lg));
  return response_mapT;

}


#if defined(ALBANY_EPETRA)
Teuchos::RCP<Epetra_Operator>
Albany::ScalarResponseFunction::
createGradientOp() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,
    "Error!  Albany::ScalarResponseFunction::createGradientOp():  " <<
    "Operator form of dg/dx is not supported for scalar responses.");
  return Teuchos::null;
}
#endif

Teuchos::RCP<Tpetra_Operator>
Albany::ScalarResponseFunction::
createGradientOpT() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,
    "Error!  Albany::ScalarResponseFunction::createGradientOp():  " <<
    "Operator form of dg/dx is not supported for scalar responses.");
  return Teuchos::null;
}

#if defined(ALBANY_EPETRA)
void
Albany::ScalarResponseFunction::
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
    dg_dx.getMultiVector().get(), dg_dxdot.getMultiVector().get(), dg_dxdotdot.getMultiVector().get(),
    dg_dp.getMultiVector().get());
}
#endif

void
Albany::ScalarResponseFunction::
evaluateDerivativeT(
  const double current_time,
  const Tpetra_Vector* xdotT,
  const Tpetra_Vector* xdotdotT,
  const Tpetra_Vector& xT,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Tpetra_Vector* gT,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
  const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{

  const Teuchos::RCP<Tpetra_MultiVector> dg_dxT =
    Teuchos::nonnull(dg_dx.getMultiVector()) ?
    ConverterT::getTpetraMultiVector(dg_dx.getMultiVector()) :
    Teuchos::null;

  const Teuchos::RCP<Tpetra_MultiVector> dg_dxdotT =
    Teuchos::nonnull(dg_dxdot.getMultiVector()) ?
    ConverterT::getTpetraMultiVector(dg_dxdot.getMultiVector()) :
    Teuchos::null;

  const Teuchos::RCP<Tpetra_MultiVector> dg_dxdotdotT =
    Teuchos::nonnull(dg_dxdotdot.getMultiVector()) ?
    ConverterT::getTpetraMultiVector(dg_dxdotdot.getMultiVector()) :
    Teuchos::null;

  const Teuchos::RCP<Tpetra_MultiVector> dg_dpT =
    Teuchos::nonnull(dg_dp.getMultiVector()) ?
    ConverterT::getTpetraMultiVector(dg_dp.getMultiVector()) :
    Teuchos::null;

  this->evaluateGradientT(
    current_time, xdotT, xdotdotT, xT, p, deriv_p, gT,
    dg_dxT.get(), dg_dxdotT.get(), dg_dxdotdotT.get(), dg_dpT.get());
}


#ifdef ALBANY_SG
void
Albany::ScalarResponseFunction::
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
    sg_g, sg_dg_dx.getMultiVector().get(), sg_dg_dxdot.getMultiVector().get(), sg_dg_dxdotdot.getMultiVector().get(),
    sg_dg_dp.getMultiVector().get());
}
#endif 
#ifdef ALBANY_ENSEMBLE 

void
Albany::ScalarResponseFunction::
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
    mp_g, mp_dg_dx.getMultiVector().get(), mp_dg_dxdot.getMultiVector().get(), mp_dg_dxdotdot.getMultiVector().get(),
    mp_dg_dp.getMultiVector().get());
}
#endif
