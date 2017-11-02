//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_SAMPLING_BASED_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_SAMPLING_BASED_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_ScalarResponseFunction.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_Quadrature.hpp"
#endif

namespace Albany {

  /*!
   * \brief Response function implementation for SG and MP functions using
   * a sampling-based scheme
   */
  class SamplingBasedScalarResponseFunction : 
    public ScalarResponseFunction {
  public:
  
    //! Default constructor
    SamplingBasedScalarResponseFunction(
      const Teuchos::RCP<const Teuchos_Comm>& commT) : 
      ScalarResponseFunction(commT) {};

    //! Destructor
    virtual ~SamplingBasedScalarResponseFunction() {};

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
      Stokhos::ProductEpetraMultiVector* mp_dg_dp);
#endif

    //@}

  private:

    //! Private to prohibit copying
    SamplingBasedScalarResponseFunction(
      const SamplingBasedScalarResponseFunction&);
    
    //! Private to prohibit copying
    SamplingBasedScalarResponseFunction& operator=(
      const SamplingBasedScalarResponseFunction&);

  protected:

  };

}

#endif // ALBANY_SAMPLING_BASED_SCALAR_RESPONSE_FUNCTION_HPP
