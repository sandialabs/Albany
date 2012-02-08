/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
    ScalarResponseFunction(const Teuchos::RCP<const Epetra_Comm>& comm_) :
      comm(comm_) {};

    //! Destructor
    virtual ~ScalarResponseFunction() {};

    //! Get the number of responses
    virtual unsigned int numResponses() const = 0;

    //! Get the comm
    virtual Teuchos::RCP<const Epetra_Comm> getComm() const {
      return comm;
    }

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void evaluateGradient(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      Epetra_MultiVector* dg_dx,
      Epetra_MultiVector* dg_dxdot,
      Epetra_MultiVector* dg_dp) = 0;

    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGGradient(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp) = 0;

    //! Evaluate multi-point derivative
    virtual void evaluateMPGradient(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_dg_dx,
      Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dp) = 0;

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
    virtual Teuchos::RCP<Epetra_Operator> createGradientOp() const;

    //! Get the map associate with this response
    virtual Teuchos::RCP<const Epetra_Map> responseMap() const;

    //! Evaluate derivative dg/dx, dg/dxdot, dg/dp
    virtual void evaluateDerivative(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      const EpetraExt::ModelEvaluator::Derivative& dg_dx,
      const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
      const EpetraExt::ModelEvaluator::Derivative& dg_dp);

    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGDerivative(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp);

    //! Evaluate multi-point derivative
    virtual void evaluateMPDerivative(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp);

    //@}


  private:

    //! Private to prohibit copying
    ScalarResponseFunction(const ScalarResponseFunction&);
    
    //! Private to prohibit copying
    ScalarResponseFunction& operator=(const ScalarResponseFunction&);

  protected:

    //! Comm for forming response map
    Teuchos::RCP<const Epetra_Comm> comm;

  };

}

#endif // ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
