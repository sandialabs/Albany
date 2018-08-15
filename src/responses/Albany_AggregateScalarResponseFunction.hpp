//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

  /*!
   * \brief A response function that aggregates together multiple response
   * functions into one.
   */
  class AggregateScalarResponseFunction : 
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    AggregateScalarResponseFunction(
      const Teuchos::RCP<const Teuchos_Comm>& commT,
      const Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> >& responses);

    //! Setup response function
    virtual void setup();

    //! Perform post registration setup
    virtual void postRegSetup();

    //! Destructor
    virtual ~AggregateScalarResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate response
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  const Teuchos::RCP<Thyra_Vector>& gT); 

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
		  const Teuchos::Array<ParamVec>& p,
		  ParamVec* deriv_p,
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
		  ParamVec* deriv_p,
		  const Teuchos::RCP<Thyra_Vector>& g,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

  private:

    //! Evaluate Multi Vector distributed derivative dg_dp
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

  private:

    //! Private to prohibit copying
    AggregateScalarResponseFunction(const AggregateScalarResponseFunction&);
    
    //! Private to prohibit copying
    AggregateScalarResponseFunction& operator=(const AggregateScalarResponseFunction&);

  protected:

    //! Response functions to aggregate
    Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> > responses;

    // We would LOVE to use this as return value for 'responseVectorSpace'. However,
    // the Epetra stack uses the EpetraExt as model evaluator base class, which has
    // ZERO knowledge about vector spaces, and instead would build a monolithic map,
    // losing all the information about the blocks anyways. However, this vs is still
    // useful to create local response blocks during the evaluation routines.
    Teuchos::RCP<const Thyra_ProductVectorSpace> productVectorSpace;
  };

}

#endif // ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP
