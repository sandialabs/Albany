//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

    //! Get the comm
    virtual Teuchos::RCP<const Teuchos_Comm> getComm() const {
      return commT;
    }

#if defined(ALBANY_EPETRA)
    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void evaluateGradient(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      Epetra_MultiVector* dg_dx,
      Epetra_MultiVector* dg_dxdot,
      Epetra_MultiVector* dg_dxdotdot,
      Epetra_MultiVector* dg_dp) = 0;
#endif

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp - Tpetra
    virtual void evaluateGradientT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* gT,
      Tpetra_MultiVector* dg_dxT,
      Tpetra_MultiVector* dg_dxdotT,
      Tpetra_MultiVector* dg_dxdotdotT,
      Tpetra_MultiVector* dg_dpT) = 0;
    

   //! Evaluate stochastic Galerkin derivative
#ifdef ALBANY_SG_MP
    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGGradient(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp) = 0;

    //! Evaluate multi-point derivative
    virtual void evaluateMPGradient(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_dg_dx,
      Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dp) = 0;
#endif //ALBANY_SG_MP

    //! \name Implementation of AbstractResponseFunction virtual methods
    //@{

#if defined(ALBANY_EPETRA)
    //! Setup response function
    virtual void setup() {}
#endif

    //! Setup response function
    virtual void setupT() {}

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
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<Epetra_Operator> createGradientOp() const;
#endif
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;

#if defined(ALBANY_EPETRA)
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Epetra_Map> responseMap() const;
#endif
    
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const;

#if defined(ALBANY_EPETRA)
    //! Evaluate derivative dg/dx, dg/dxdot, dg/dp
    virtual void evaluateDerivative(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      const EpetraExt::ModelEvaluator::Derivative& dg_dx,
      const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
      const EpetraExt::ModelEvaluator::Derivative& dg_dxdotdot,
      const EpetraExt::ModelEvaluator::Derivative& dg_dp);
#endif

    virtual void evaluateDerivativeT(
      const double current_time,
      const Tpetra_Vector* xdot,
      const Tpetra_Vector* xdotdot,
      const Tpetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* g,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp);
    
#ifdef ALBANY_SG_MP
    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGDerivative(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdotdot,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp);

    //! Evaluate multi-point derivative
    virtual void evaluateMPDerivative(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdotdot,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp);
#endif //ALBANY_SG_MP

    //@}


  private:

    //! Private to prohibit copying
    ScalarResponseFunction(const ScalarResponseFunction&);
    
    //! Private to prohibit copying
    ScalarResponseFunction& operator=(const ScalarResponseFunction&);

  protected:

    //! Comm for forming response map
    Teuchos::RCP<const Teuchos_Comm> commT;

  };

}

#endif // ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
