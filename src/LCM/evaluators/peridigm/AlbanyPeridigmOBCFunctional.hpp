//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP
#define ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP

#include "Albany_ScalarResponseFunction.hpp"

namespace Albany {
/** 
 * \brief Response Description
 */
  class AlbanyPeridigmOBCFunctional : 
      public ScalarResponseFunction
  {
  public:
    AlbanyPeridigmOBCFunctional(const Teuchos::RCP<const Teuchos_Comm>& commT);

    //! Destructor
    virtual ~AlbanyPeridigmOBCFunctional();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void
    evaluateResponseT(const double current_time,
         const Tpetra_Vector* xdotT,
         const Tpetra_Vector* xdotdotT,
         const Tpetra_Vector& xT,
         const Teuchos::Array<ParamVec>& p,
         Tpetra_Vector& gT);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangentT(const double alpha, 
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

#ifdef ALBANY_EPETRA
    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void
    evaluateGradient(const double current_time,
         const Epetra_Vector* xdot,
         const Epetra_Vector* xdotdot,
         const Epetra_Vector& x,
         const Teuchos::Array<ParamVec>& p,
         ParamVec* deriv_p,
         Epetra_Vector* g,
         Epetra_MultiVector* dg_dx,
         Epetra_MultiVector* dg_dxdot,
         Epetra_MultiVector* dg_dxdotdot,
         Epetra_MultiVector* dg_dp);
#endif

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void 
    evaluateGradientT(const double current_time,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector* xdotdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Tpetra_Vector* gT,
		     Tpetra_MultiVector* dg_dxT,
		     Tpetra_MultiVector* dg_dxdotT,
		     Tpetra_MultiVector* dg_dxdotdotT,
		     Tpetra_MultiVector* dg_dpT);

#ifdef ALBANY_EPETRA
    //! Evaluate distributed parameter derivative dg/dp
    virtual void
    evaluateDistParamDeriv(
             const double current_time,
             const Epetra_Vector* xdot,
             const Epetra_Vector* xdotdot,
             const Epetra_Vector& x,
             const Teuchos::Array<ParamVec>& param_array,
             const std::string& dist_param_name,
             Epetra_MultiVector* dg_dp);
#endif

    //! \name Stochastic Galerkin evaluation functions
    //@{
#ifdef ALBANY_SG
    //! Intialize stochastic Galerkin method
    virtual void init_sg(
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double>>& basis,
      const Teuchos::RCP<const Stokhos::Quadrature<int,double>>& quad,
      const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double>>& expansion,
      const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm) {}

    //! Evaluate stochastic Galerkin response
    virtual void evaluateSGResponse(
      const double curr_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType>>& sg_p_vals,
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
      const Teuchos::Array< Teuchos::Array<SGType>>& sg_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_gp);

    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGGradient(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType>>& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp);
#endif

    //@}

#ifdef ALBANY_ENSEMBLE 
    //! \name Multi-point evaluation functions
    //@{

    //! Evaluate multi-point response functions
    virtual void evaluateMPResponse(
      const double curr_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType>>& mp_p_vals,
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
      const Teuchos::Array< Teuchos::Array<MPType>>& mp_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_JV,
      Stokhos::ProductEpetraMultiVector* mp_gp);

    //! Evaluate multi-point derivative
    virtual void evaluateMPGradient(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType>>& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_dg_dx,
      Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dp);
#endif

    //@}
  private:

    //! Private to prohibit copying
    AlbanyPeridigmOBCFunctional(const AlbanyPeridigmOBCFunctional&);

     //! Private to prohibit copying
    AlbanyPeridigmOBCFunctional& operator=(const AlbanyPeridigmOBCFunctional&);
  };
	
}

#endif
