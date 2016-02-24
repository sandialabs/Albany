//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: Epetra ifdef'ed out except SG and MP when ALBANY_EPETRA_EXE is off.

#ifndef ALBANY_KL_RESPONSE_FUNCTION_HPP
#define ALBANY_KL_RESPONSE_FUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_Application.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_FancyOStream.hpp"

#if defined(ALBANY_EPETRA)
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include "Epetra_CrsGraph.h"
#endif

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

#if defined(ALBANY_EPETRA)
    //! Setup response function
    virtual void setup() { response->setup(); }
#endif
    //! Setup response function
    virtual void setupT() { response->setupT(); }

#if defined(ALBANY_EPETRA)
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Epetra_Map> responseMap() const;
#endif
    
    //! Get the map associate with this response - Tpetra version
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const;

    /*! 
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const;

#if defined(ALBANY_EPETRA)
    //! Create operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Epetra_Operator> createGradientOp() const;
#endif
    //! Create Tpetra operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses 
    virtual void evaluateResponseT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& p,
      Tpetra_Vector& gT);
    
    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
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
      Tpetra_MultiVector* gp);

#if defined(ALBANY_EPETRA)
    //! Evaluate distributed parameter derivative dg/dp
    virtual void evaluateDistParamDeriv(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Epetra_MultiVector*  dg_dp);
#endif

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
#if defined(ALBANY_EPETRA)
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
    //@}

    //! \name Stochastic Galerkin evaluation functions
    //@{

#ifdef ALBANY_SG
    //! Intialize stochastic Galerkin method
    virtual void init_sg(
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
      const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
      const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
      const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm);

    //! Evaluate stochastic Galerkin response
    virtual void evaluateSGResponse(
      const double curr_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly& sg_g);

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
      Stokhos::EpetraMultiVectorOrthogPoly* sg_gp);

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
      Stokhos::ProductEpetraVector& mp_g);

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
      Stokhos::ProductEpetraMultiVector* mp_gp);

    //! Evaluate stochastic Galerkin derivative
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
#endif

    //@}

  private:

    //! Private to prohibit copying
    KLResponseFunction(const KLResponseFunction&);
    
    //! Private to prohibit copying
    KLResponseFunction& operator=(const KLResponseFunction&);

  protected:

#if defined(ALBANY_EPETRA)
    bool computeKL(const Stokhos::EpetraVectorOrthogPoly& sg_u,
		   const int NumKL,
		   Teuchos::Array<double>& evals,
		   Teuchos::RCP<Epetra_MultiVector>& evecs);
#endif

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

}

#endif // ALBANY_KL_RESPONSE_FUNCTION_HPP
