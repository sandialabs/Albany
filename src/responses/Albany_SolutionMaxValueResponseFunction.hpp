//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTIONMAXVALUERESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONMAXVALUERESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class SolutionMaxValueResponseFunction : 
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    SolutionMaxValueResponseFunction(
      const Teuchos::RCP<const Epetra_Comm>& comm, 
      int neq = 1, int eq = 0, bool interleavedOrdering=true);

    //! Destructor
    virtual ~SolutionMaxValueResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
		    const double beta,
		    const double current_time,
		    bool sum_derivs,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    ParamVec* deriv_p,
		    const Epetra_MultiVector* Vxdot,
		    const Epetra_MultiVector* Vx,
		    const Epetra_MultiVector* Vp,
		    Epetra_Vector* g,
		    Epetra_MultiVector* gx,
		    Epetra_MultiVector* gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void 
    evaluateGradient(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Epetra_Vector* g,
		     Epetra_MultiVector* dg_dx,
		     Epetra_MultiVector* dg_dxdot,
		     Epetra_MultiVector* dg_dp);

  private:

    //! Private to prohibit copying
    SolutionMaxValueResponseFunction(const SolutionMaxValueResponseFunction&);
    
    //! Private to prohibit copying
    SolutionMaxValueResponseFunction& operator=(const SolutionMaxValueResponseFunction&);

  protected:

    //! Number of equations per node
    int neq;

    //! Equation we want to get the max value from
    int eq;

    //! Flag for interleaved verus blocked unknown ordering
    bool interleavedOrdering;

    //! Compute max value and index
    void computeMaxValue(const Epetra_Vector& x, double& val, int& index);

  };

}

#endif // ALBANY_SOLUTIONMAXVALUERESPONSEFUNCTION_HPP
