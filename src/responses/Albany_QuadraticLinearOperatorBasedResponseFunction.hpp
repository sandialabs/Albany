//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_QUADRATICLINEAROPERATORBASEDRESPONSEFUNCTION_HPP
#define ALBANY_QUADRATICLINEAROPERATORBASEDRESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Albany_Utils.hpp"

namespace Albany {
  class Application;

  /*!
   * \brief Response function computing the scalar:
   * coeff p' A' inv(D) A p,
   * for a field parameter p, a matrix A and a diagonal matrix D.
   * The matrices A and D are loaded from "*.mm" ASCII files.
   */
  class QuadraticLinearOperatorBasedResponseFunction :
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    QuadraticLinearOperatorBasedResponseFunction(
        const Teuchos::RCP<const Application> &app,
        Teuchos::ParameterList &responseParams);

    //! Destructor
    virtual ~QuadraticLinearOperatorBasedResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    void loadLinearOperator();

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
      const Teuchos::RCP<Thyra_Vector>& g);


    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
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
      const Teuchos::RCP<Thyra_MultiVector>& gp);
    
    virtual void 
    evaluateGradient(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      const int parameter_index,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

  void printResponse(Teuchos::RCP<Teuchos::FancyOStream> out);

  private:

    //! Evaluate distributed parameter derivative = dg/dp
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    virtual void
    evaluate_HessVecProd_xx(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_xp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_px(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_pp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

  private:

    //! Private to prohibit copying
    QuadraticLinearOperatorBasedResponseFunction(const QuadraticLinearOperatorBasedResponseFunction&);
    
    QuadraticLinearOperatorBasedResponseFunction& operator=(const QuadraticLinearOperatorBasedResponseFunction&);

    Teuchos::RCP<const Application> app_;
    Teuchos::RCP<Thyra_Vector> D_,g_,vec1_,vec2_;
    Teuchos::RCP<Thyra_LinearOp> A_;
    std::string field_name_, file_name_A_, file_name_D_;
    double coeff_;
  };

} // namespace Albany

#endif // ALBANY_SOLUTIONTWONORMRESPONSEFUNCTION_HPP
