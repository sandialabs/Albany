//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTION_MIN_VALUE_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_MIN_VALUE_RESPONSE_FUNCTION_HPP

#include "Albany_MeshSpecs.hpp"
#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

/*!
 * \brief Reponse function representing the min of the solution values
 */
class SolutionMinValueResponseFunction : public SamplingBasedScalarResponseFunction
{
public:
  //! Default constructor
  SolutionMinValueResponseFunction(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    int neq = 1, int eq = 0, DiscType interleavedOrdering = DiscType::Interleaved);

  //! Destructor
  ~SolutionMinValueResponseFunction() = default;

  //! Get the number of responses
  unsigned int numResponses() const { return 1; }

  //! Evaluate responses
  void evaluateResponse(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g);

  //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  void evaluateTangent(const double alpha,
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
    const Teuchos::RCP<Thyra_MultiVector>& gp);

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  void evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

  //! Evaluate distributed parameter derivative dg/dp
  void evaluateDistParamDeriv(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

protected:

  //! Compute min value
  void computeMinValue(const Teuchos::RCP<const Thyra_Vector>& x, ST& val);

  //! Number of equations per node
  int neq;

  //! Equation we want to get the max value from
  int eq;

  Teuchos::RCP<const Teuchos_Comm> comm_;

  //! Flag for interleaved verus blocked unknown ordering
  DiscType interleavedOrdering;
};

} // namespace Albany

#endif // ALBANY_SOLUTION_MIN_VALUE_RESPONSE_FUNCTION_HPP
