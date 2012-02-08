/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Albany_ScalarResponseFunction.hpp"
#include "Teuchos_TestForException.hpp"
#include "Epetra_LocalMap.h"

Teuchos::RCP<const Epetra_Map> 
Albany::ScalarResponseFunction::
responseMap() const
{
  int num_responses = this->numResponses();
  Teuchos::RCP<const Epetra_LocalMap> response_map =
    Teuchos::rcp(new Epetra_LocalMap(num_responses, 0, *comm));
  return response_map;
}

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

void
Albany::ScalarResponseFunction::
evaluateDerivative(
  const double current_time,
  const Epetra_Vector* xdot,
  const Epetra_Vector& x,
  const Teuchos::Array<ParamVec>& p,
  ParamVec* deriv_p,
  Epetra_Vector* g,
  const EpetraExt::ModelEvaluator::Derivative& dg_dx,
  const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
  const EpetraExt::ModelEvaluator::Derivative& dg_dp)
{
  this->evaluateGradient(
    current_time, xdot, x, p, deriv_p, g,
    dg_dx.getMultiVector().get(), dg_dxdot.getMultiVector().get(),
    dg_dp.getMultiVector().get());
}

void 
Albany::ScalarResponseFunction::
evaluateSGDerivative(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
  const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp)
{
  this->evaluateSGGradient(
    current_time, sg_xdot, sg_x, p, sg_p_index, sg_p_vals, deriv_p,
    sg_g, sg_dg_dx.getMultiVector().get(), sg_dg_dxdot.getMultiVector().get(),
    sg_dg_dp.getMultiVector().get());
}

void 
Albany::ScalarResponseFunction::
evaluateMPDerivative(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp)
{
  this->evaluateMPGradient(
    current_time, mp_xdot, mp_x, p, mp_p_index, mp_p_vals, deriv_p,
    mp_g, mp_dg_dx.getMultiVector().get(), mp_dg_dxdot.getMultiVector().get(),
    mp_dg_dp.getMultiVector().get());
}
