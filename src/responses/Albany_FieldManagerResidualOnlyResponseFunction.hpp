//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP
#define ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP

#include "Albany_FieldManagerScalarResponseFunction.hpp"

/* todo
   - check that Residual is the only version called for all overridden methods.
 */

namespace Albany {

  /*!
   * \brief Reponse function that calls an evaluator that implements only EvalT=
   * PHAL::AlbanyTraits::Residual.
   *
   * It seems that a common use case for a response function is to do something
   * with solution data and data available in evaluator worksets, but not
   * necessarily to implement a mathematical function g whose derivatives can be
   * formed. Examples including transferring data to another module in a loose
   * coupling of Albany with other software, or writing special files.
   *   This Response Function calls only EvalT=PHAL::AlbanyTraits::Residual
   * forms of overridden methods. It returns 0 for all derivatives. Hence a
   * sensitivity will turn out to be 0.
   */
  class FieldManagerResidualOnlyResponseFunction : 
    public FieldManagerScalarResponseFunction {
  public:
  
    //! Constructor
    FieldManagerResidualOnlyResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      const Teuchos::RCP<Albany::AbstractProblem>& problem,
      const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
      const Teuchos::RCP<Albany::StateManager>& stateMgr,
      Teuchos::ParameterList& responseParams);

    //! Does not actually compute the tangent; just calls evaluateResponseT
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

    //! Does not actually compute the gradient; just calls evaluateResponseT
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

  private:

    //! Private to prohibit copying
    FieldManagerResidualOnlyResponseFunction(
      const FieldManagerResidualOnlyResponseFunction&);
    FieldManagerResidualOnlyResponseFunction& operator=(
      const FieldManagerResidualOnlyResponseFunction&);

  };

}

#endif // ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP
