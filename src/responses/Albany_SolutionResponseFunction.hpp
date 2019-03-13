//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP

#include "Albany_DistributedResponseFunction.hpp"
#include "Albany_ThyraCrsGraphProxy.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

class Application;

/*!
 * \brief A response function given by (possibly a portion of) the solution
 */
class SolutionResponseFunction : public DistributedResponseFunction {
public:

  //! Default constructor
  SolutionResponseFunction(const Teuchos::RCP<Albany::Application>& application,
                           const Teuchos::ParameterList& responseParams);

  //! Destructor
  virtual ~SolutionResponseFunction() = default;

  //! Setup response function
  void setup() override;
  
  //! Get the map associate with this response
  Teuchos::RCP<const Thyra_VectorSpace> responseVectorSpace() const override { return culled_vs; }

  //! Create operator for gradient
  Teuchos::RCP<Thyra_LinearOp> createGradientOp() const override;

  //! \name Deterministic evaluation functions
  //@{

  //! Evaluate responses
  void evaluateResponse(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g) override;
  
  //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  void evaluateTangent(
    const double alpha, 
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
    const Teuchos::RCP<Thyra_MultiVector>& gp) override;

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp - Tpetra
  void evaluateGradient(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_LinearOp>& dg_dx,
    const Teuchos::RCP<Thyra_LinearOp>& dg_dxdot,
    const Teuchos::RCP<Thyra_LinearOp>& dg_dxdotdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp) override;

  //! Evaluate distributed parameter derivative = dg/dp
  void evaluateDistParamDeriv(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp) override;
  //@}

protected:
  
  //Tpetra version of above function
  void cullSolution(const Teuchos::RCP<const Thyra_MultiVector>& x, 
                    const Teuchos::RCP<      Thyra_MultiVector>& x_culled) const;

  //! Mask for DOFs to keep
  Teuchos::Array<bool> keepDOF;
  int numKeepDOF;
  

  //! Vector space for response
  Teuchos::RCP<const Thyra_SpmdVectorSpace> solution_vs;
  Teuchos::RCP<const Thyra_SpmdVectorSpace> culled_vs;

  //! The restriction operator, use to cull the solution
  Teuchos::RCP<Thyra_LinearOp> cull_op;

  //! Graph of gradient operator
  // Note: not all concrete implementations of Thyra_LinearOp implement the clone() method,
  //       so we need to create the gradient op every time createGradientOp is called.
  Teuchos::RCP<ThyraCrsGraphProxy> graph_proxy;

};

} // namespace Albany

#endif // ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
