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
#ifdef ALBANY_STOKHOS
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Stokhos_EpetraMultiVectorOrthogPoly.hpp"
#include "Stokhos_ProductEpetraVector.hpp"
#include "Stokhos_ProductEpetraMultiVector.hpp"
#endif
#endif
#ifdef ALBANY_STOKHOS
#include "Stokhos_OrthogPolyBasis.hpp"
#include "Stokhos_OrthogPolyExpansion.hpp"
#include "Stokhos_Quadrature.hpp"
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
    
#if defined(ALBANY_EPETRA)
    //! Evaluate distributed parameter derivative dg/dp
    virtual void evaluateDistParamDeriv(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Epetra_MultiVector*  dg_dp) = 0;
#endif
    //@}

    //! \name Stochastic Galerkin evaluation functions
    //@{

#ifdef ALBANY_SG
    //! Intialize stochastic Galerkin method
    virtual void init_sg(
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
      const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
      const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
      const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm) = 0;

    //! Evaluate stochastic Galerkin response
    virtual void evaluateSGResponse(
      const double curr_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly& sg_g) = 0;

    //! Evaluate stochastic Galerkin tangent
    virtual void evaluateSGTangent(
      const double alpha, 
      const double beta, 
      const double omega, 
      const double current_time,
      bool sum_derivs,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_gp) = 0;

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
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp) = 0;
#endif

    //@}

    //! \name Multi-point evaluation functions
    //@{

#ifdef ALBANY_ENSEMBLE 
    //! Evaluate multi-point response functions
    virtual void evaluateMPResponse(
      const double curr_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      Stokhos::ProductEpetraVector& mp_g) = 0;

    //! Evaluate multi-point tangent
    virtual void evaluateMPTangent(
      const double alpha, 
      const double beta, 
      const double omega, 
      const double current_time,
      bool sum_derivs,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_JV,
      Stokhos::ProductEpetraMultiVector* mp_gp) = 0;

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
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp) = 0;
#endif

    //@}

  private:

    //! Private to prohibit copying
    AbstractResponseFunction(const AbstractResponseFunction&);
    
    //! Private to prohibit copying
    AbstractResponseFunction& operator=(const AbstractResponseFunction&);

  };

}

#endif // ALBANY_ABSTRACTRESPONSEFUNCTION_HPP
