//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_ScalarResponseFunction.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"

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

Teuchos::RCP<Tpetra_Operator>
Albany::ScalarResponseFunction::
createGradientOpT() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,
    "Error!  Albany::ScalarResponseFunction::createGradientOpT():  " <<
    "Operator form of dg/dx is not supported for scalar responses.");
  return Teuchos::null;
}

void
Albany::ScalarResponseFunction::
evaluateDerivative(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{
  this->evaluateGradient(
    current_time, x, xdot, xdotdot, p, deriv_p, g,
    dg_dx.getMultiVector(), dg_dxdot.getMultiVector(),
    dg_dxdotdot.getMultiVector(), dg_dp.getMultiVector());
}
