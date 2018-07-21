//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_KL_RESPONSE_FUNCTION_HPP
#define ALBANY_KL_RESPONSE_FUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_Application.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_FancyOStream.hpp"

namespace Albany {

  /*!
   * \brief A response function given by the KL decomposition of another
   * response function.
   * 
   * It only defines the SG methods.
   */
  class KLResponseFunction : public AbstractResponseFunction {
  public:
  
    //! Default constructor
    KLResponseFunction(
      const Teuchos::RCP<AbstractResponseFunction>& response,
      Teuchos::ParameterList& responseParams);

    //! Destructor
    virtual ~KLResponseFunction();

    //! Setup response function
    virtual void setup() { response->setup(); }

    //! Perform post registration setup (do nothing)
    virtual void postRegSetup(){};
    
    //! Get the map associate with this response - Tpetra version
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const;

    /*! 
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const;

    //! Create Tpetra operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses 
    virtual void evaluateResponse(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      Tpetra_Vector& gT);
    
    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void evaluateTangent(
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
      Tpetra_Vector* g,
      Tpetra_MultiVector* gx,
      Tpetra_MultiVector* gp);

    //! Evaluate distributed parameter derivative dg/dp
    virtual void evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Tpetra_MultiVector*  dg_dpT);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void evaluateDerivative(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* g,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp);
    //@}

  private:

    //! Private to prohibit copying
    KLResponseFunction(const KLResponseFunction&);
    
    //! Private to prohibit copying
    KLResponseFunction& operator=(const KLResponseFunction&);

  protected:

    //! Response function we work with
    Teuchos::RCP<AbstractResponseFunction> response;

    //! Response parameters
    Teuchos::ParameterList responseParams;

    //! Output stream;
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Number of KL terms
    int num_kl;

  };

} // namespace Albany

#endif // ALBANY_KL_RESPONSE_FUNCTION_HPP
