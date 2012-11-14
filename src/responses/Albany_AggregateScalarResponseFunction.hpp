//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
      const Teuchos::RCP<const Epetra_Comm>& comm,
      const Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> >& responses);

    //! Destructor
    virtual ~AggregateScalarResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g);
    
    //! Evaluate responses - Tpetra
    virtual void 
    evaluateResponseT(const double current_time,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& p,
		     Tpetra_Vector& gT); 

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
		    const double beta,
		    const double current_time,
		    bool sum_derivs,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    ParamVec* deriv_p,
		    const Epetra_MultiVector* Vxdot,
		    const Epetra_MultiVector* Vx,
		    const Epetra_MultiVector* Vp,
		    Epetra_Vector* g,
		    Epetra_MultiVector* gx,
		    Epetra_MultiVector* gp);

    virtual void 
    evaluateTangentT(const double alpha, 
		    const double beta,
		    const double current_time,
		    bool sum_derivs,
		    const Tpetra_Vector* xdot,
		    const Tpetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    ParamVec* deriv_p,
		    const Tpetra_MultiVector* Vxdot,
		    const Tpetra_MultiVector* Vx,
		    const Tpetra_MultiVector* Vp,
		    Tpetra_Vector* g,
		    Tpetra_MultiVector* gx,
		    Tpetra_MultiVector* gp);
   
    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void 
    evaluateGradient(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Epetra_Vector* g,
		     Epetra_MultiVector* dg_dx,
		     Epetra_MultiVector* dg_dxdot,
		     Epetra_MultiVector* dg_dp);
    
    virtual void 
    evaluateGradientT(const double current_time,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Tpetra_Vector* gT,
		     Tpetra_MultiVector* dg_dxT,
		     Tpetra_MultiVector* dg_dxdotT,
		     Tpetra_MultiVector* dg_dpT);

  private:

  private:

    //! Private to prohibit copying
    AggregateScalarResponseFunction(const AggregateScalarResponseFunction&);
    
    //! Private to prohibit copying
    AggregateScalarResponseFunction& operator=(const AggregateScalarResponseFunction&);

  protected:

    //! Response functions to aggregate
    Teuchos::Array< Teuchos::RCP<ScalarResponseFunction> > responses;

  };

}

#endif // ALBANY_AGGREGATE_SCALAR_RESPONSE_FUNCTION_HPP
