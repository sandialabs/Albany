//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_RESPONSE_FUNCTION_HPP
#define ALBANY_ABSTRACT_RESPONSE_FUNCTION_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Thyra_ModelEvaluatorBase.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_DataTypes.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a response function
   */
  class AbstractResponseFunction {
  public:

    //! Default constructor
    AbstractResponseFunction() {};

    //! Destructor
    virtual ~AbstractResponseFunction() {};

    //! Setup response function
    virtual void setup() = 0;

    //! Get the vector space associated with this response.
    virtual Teuchos::RCP<const Thyra_VectorSpace> responseVectorSpace() const = 0;

    /*!
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const = 0;

    //! Create Thyra operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Thyra_LinearOp> createGradientOp() const = 0;

    //! perform post registration setup
    virtual void postRegSetup() = 0;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses
    virtual void evaluateResponse(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::RCP<Thyra_Vector>& g) = 0;

    virtual void evaluateTangent(
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      Teuchos::Array<ParamVec>& p,
      const int parameter_index,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& gx,
      const Teuchos::RCP<Thyra_MultiVector>& gp) = 0;

   virtual void evaluateDerivative(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      const int parameter_index,
      const Teuchos::RCP<Thyra_Vector>& gT,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp) = 0;

    //! Evaluate distributed parameter derivative dg/dp
    virtual void evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp) = 0;

    //! Evaluate distributed parameter derivative dg/dp
    virtual void evaluate_HessVecProd_xx(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp) = 0;

    virtual void evaluate_HessVecProd_xp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp) = 0;

    virtual void evaluate_HessVecProd_px(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp) = 0;

    virtual void evaluate_HessVecProd_pp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp) = 0;

    //! Returns a linear operator for the Hessian of the response with respect of the parameter param_name
    virtual Teuchos::RCP<Thyra_LinearOp> get_Hess_pp_operator(const std::string& param_name) {return Teuchos::null;}
    //@}

    virtual void printResponse(
      Teuchos::RCP<Teuchos::FancyOStream> out
    ) = 0;
  private:

    //! Private to prohibit copying
    AbstractResponseFunction(const AbstractResponseFunction&);

    //! Private to prohibit copying
    AbstractResponseFunction& operator=(const AbstractResponseFunction&);

  };

} // namespace Albany

#endif // ALBANY_ABSTRACT_RESPONSE_FUNCTION_HPP
