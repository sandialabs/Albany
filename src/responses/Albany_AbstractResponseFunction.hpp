//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: Epetra ifdef'ed out if ALBANY_EPETRA_EXE is off, except SG and MP.

#ifndef ALBANY_ABSTRACTRESPONSEFUNCTION_HPP
#define ALBANY_ABSTRACTRESPONSEFUNCTION_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "EpetraExt_ModelEvaluator.h"
#include "EpetraExt_MultiComm.h"
#endif
#include "PHAL_AlbanyTraits.hpp"
#include "Thyra_ModelEvaluatorBase.hpp"
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

#if defined(ALBANY_EPETRA)
    //! Setup response function
    virtual void setup() = 0;
#endif

    //! Setup response function
    virtual void setupT() = 0;

#if defined(ALBANY_EPETRA)
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Epetra_Map> responseMap() const = 0;
#endif    

    //! Get the map associate with this response - Tpetra version 
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const = 0;

    /*! 
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const = 0;

#if defined(ALBANY_EPETRA)
    //! Create operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Epetra_Operator> createGradientOp() const = 0;
#endif
    //! Create Tpetra operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const = 0;

    //! perform post registration setup
    virtual void postRegSetup() = 0;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses

    virtual void evaluateResponseT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& p,
      Tpetra_Vector& gT) = 0;
    
    virtual void evaluateTangentT(
      const double alpha, 
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Tpetra_Vector* xdot,
      const Tpetra_Vector* xdotdot,
      const Tpetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Tpetra_MultiVector* Vxdot,
      const Tpetra_MultiVector* Vxdotdot,
      const Tpetra_MultiVector* Vx,
      const Tpetra_MultiVector* Vp,
      Tpetra_Vector* g,
      Tpetra_MultiVector* gx,
      Tpetra_MultiVector* gp) = 0;

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
      const EpetraExt::ModelEvaluator::Derivative& dg_dp) = 0;
#endif

   virtual void evaluateDerivativeT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* gT,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp) = 0;
    
//! Evaluate distributed parameter derivative dg/dp
    virtual void evaluateDistParamDerivT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Tpetra_MultiVector*  dg_dpT) = 0;
    //@}

  private:

    //! Private to prohibit copying
    AbstractResponseFunction(const AbstractResponseFunction&);
    
    //! Private to prohibit copying
    AbstractResponseFunction& operator=(const AbstractResponseFunction&);

  };

}

#endif // ALBANY_ABSTRACTRESPONSEFUNCTION_HPP
