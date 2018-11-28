//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_SOLUTIONAVERAGERESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONAVERAGERESPONSEFUNCTION_HPP

#include "Albany_ScalarResponseFunction.hpp"

namespace Albany {

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class SolutionAverageResponseFunction : 
    public ScalarResponseFunction {
  public:
  
    //! Default constructor
    SolutionAverageResponseFunction(
      const Teuchos::RCP<const Teuchos_Comm>& commT);

    //! Destructor
    virtual ~SolutionAverageResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  const Teuchos::RCP<Thyra_Vector>& g);


    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
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

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
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

    //! Evaluate distributed parameter derivative dg/dp
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

    void evaluateResponseImpl (const Thyra_Vector& x,
		                           Thyra_Vector& g);

    //! Private to prohibit copying
    SolutionAverageResponseFunction(const SolutionAverageResponseFunction&);
    
    //! Private to prohibit copying
    SolutionAverageResponseFunction& operator=(const SolutionAverageResponseFunction&);

    Teuchos::RCP<Thyra_Vector>      one;
    Teuchos::RCP<Thyra_MultiVector> ones;

  };

} // namespace Albany

#endif // ALBANY_SOLUTIONAVERAGERESPONSEFUNCTION_HPP
