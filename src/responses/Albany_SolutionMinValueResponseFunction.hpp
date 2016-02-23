//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_SOLUTIONMINVALUERESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONMINVALUERESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class SolutionMinValueResponseFunction :
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    SolutionMinValueResponseFunction(
      const Teuchos::RCP<const Teuchos_Comm>& commT, 
      int neq = 1, int eq = 0, bool interleavedOrdering=true);

    //! Destructor
    virtual ~SolutionMinValueResponseFunction();

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

#if defined(ALBANY_EPETRA)
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
    
    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp - Tpetra version
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

#if defined(ALBANY_EPETRA)
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

  private:

    //! Private to prohibit copying
    SolutionMinValueResponseFunction(const SolutionMinValueResponseFunction&);
    
    //! Private to prohibit copying
    SolutionMinValueResponseFunction& operator=(const SolutionMinValueResponseFunction&);

  protected:

    //! Number of equations per node
    int neq;

    //! Equation we want to get the max value from
    int eq;

    Teuchos::RCP<const Teuchos_Comm> commT_; 

    //! Flag for interleaved verus blocked unknown ordering
    bool interleavedOrdering;

#if defined(ALBANY_EPETRA)
    /*
    //! Compute max value and index
    void computeMaxValue(const Epetra_Vector& x, double& val, int& index);
    */
#endif
    //! Compute max value and index - Tpetra
    void computeMinValueT(const Tpetra_Vector& xT, double& val, int& index);

  };

}

#endif // ALBANY_SOLUTIONMINVALUERESPONSEFUNCTION_HPP
