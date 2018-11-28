//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"

namespace Albany {

  /*!
   * \brief Interface for scalar response functions
   *
   * Implements a few methods of AbstractResponseFunction specifically for
   * "scalar" valued responses, i.e., those that just return a few values.
   * In this case, the derivative is a multi-vector and the response map
   * is simpler.
   */
  class ScalarResponseFunction : 
    public AbstractResponseFunction {
  public:
  
    //! Default constructor
    ScalarResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT_) :
      commT(commT_) {};

    //! Destructor
    virtual ~ScalarResponseFunction() {};

    //! Get the number of responses
    virtual unsigned int numResponses() const = 0;

    //! Performs post registration setup
    virtual void postRegSetup(){};
    
    //! Get the comm
    virtual Teuchos::RCP<const Teuchos_Comm> getComm() const {
      return commT;
    }

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp - Tpetra
    virtual void evaluateGradient(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp) = 0;


    //! \name Implementation of AbstractResponseFunction virtual methods
    //@{

    //! Setup response function
    virtual void setup() {}

    /*! 
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const { return true; }
    
    //! Create operator for gradient
    /*!
     * Here we just throw an error.  We could actually support this a coupled
     * of ways if we wanted to.
     */
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;
    
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const;

    virtual void evaluateDerivative(
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
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp);


  private:

    //! Private to prohibit copying
    ScalarResponseFunction(const ScalarResponseFunction&);
    
    //! Private to prohibit copying
    ScalarResponseFunction& operator=(const ScalarResponseFunction&);

  protected:

    //! Comm for forming response map
    Teuchos::RCP<const Teuchos_Comm> commT;

  };

} // namespace Albany

#endif // ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
