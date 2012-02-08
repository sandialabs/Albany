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


#ifndef ALBANY_DISTRIBUTED_RESPONSE_FUNCTION_HPP
#define ALBANY_DISTRIBUTED_RESPONSE_FUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Stokhos_ProductEpetraOperator.hpp"
#include "Stokhos_EpetraOperatorOrthogPoly.hpp"

namespace Albany {

  /*!
   * \brief Interface for distributed response functions
   *
   * Implements a few methods of AbstractResponseFunction specifically for
   * distributred responses, i.e., those that involve a distributed map.
   */
  class DistributedResponseFunction : 
    public AbstractResponseFunction {
  public:
  
    //! Default constructor
    DistributedResponseFunction() {};

    //! Destructor
    virtual ~DistributedResponseFunction() {};

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void evaluateGradient(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      Epetra_Operator* dg_dx,
      Epetra_Operator* dg_dxdot,
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
      Stokhos::EpetraOperatorOrthogPoly* sg_dg_dx,
      Stokhos::EpetraOperatorOrthogPoly* sg_dg_dxdot,
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
      Stokhos::ProductEpetraOperator* mp_dg_dx,
      Stokhos::ProductEpetraOperator* mp_dg_dxdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dp) = 0;

    //! \name Implementation of AbstractResponseFunction virtual methods
    //@{

    /*! 
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const { return false; }

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
    DistributedResponseFunction(const DistributedResponseFunction&);
    
    //! Private to prohibit copying
    DistributedResponseFunction& operator=(const DistributedResponseFunction&);

  protected:

    //! Comm for forming response map
    Teuchos::RCP<const Epetra_Comm> comm;

  };

}

#endif // ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
