//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"

#include "Albany_ScalarResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"

namespace Albany {

Teuchos::RCP<const Thyra_VectorSpace>
ScalarResponseFunction::responseVectorSpace() const
{
  int num_responses = this->numResponses();
  return createLocallyReplicatedVectorSpace(num_responses,comm);
}

Teuchos::RCP<Thyra_LinearOp>
ScalarResponseFunction::createGradientOp() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,
    "Error!  Albany::ScalarResponseFunction::createGradientOpT():  " <<
    "Operator form of dg/dx is not supported for scalar responses.");
  return Teuchos::null;
}

void ScalarResponseFunction::evaluateDerivative(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    const int parameter_index,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
    const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp)
{
  this->evaluateGradient(
    current_time, x, xdot, xdotdot, p, parameter_index, g,
    dg_dx.getMultiVector(), dg_dxdot.getMultiVector(),
    dg_dxdotdot.getMultiVector(), dg_dp.getMultiVector());
}

void
ScalarResponseFunction::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> /* out */)
{
}

} // namespace Albany
