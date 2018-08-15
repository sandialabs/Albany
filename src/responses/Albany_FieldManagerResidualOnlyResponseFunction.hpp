//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP
#define ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP

#include "Albany_FieldManagerScalarResponseFunction.hpp"

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

    virtual void 
    evaluateTangent(const double alpha, 
		  const double beta,
		  const double omega,
		  const double current_time,
		  bool sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  ParamVec* deriv_p,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& gx,
      const Teuchos::RCP<Thyra_MultiVector>& gp);
    
    virtual void 
    evaluateGradient(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  ParamVec* deriv_p,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

  private:

    //! Private to prohibit copying
    FieldManagerResidualOnlyResponseFunction(
      const FieldManagerResidualOnlyResponseFunction&);
    FieldManagerResidualOnlyResponseFunction& operator=(
      const FieldManagerResidualOnlyResponseFunction&);

  };

} // namespace Albany

#endif // ALBANY_FIELD_MANAGER_RESIDUAL_ONLY_RESPONSE_FUNCTION_HPP
