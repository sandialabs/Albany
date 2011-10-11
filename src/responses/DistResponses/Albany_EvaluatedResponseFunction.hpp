/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_EVALUATEDRESPONSEFUNCTION_HPP
#define ALBANY_EVALUATEDRESPONSEFUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include "Epetra_Vector.h"
#include "EpetraExt_MultiComm.h"

namespace Albany {

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class EvaluatedResponseFunction : public AbstractResponseFunction {
  public:
  
    //! Default constructor
    EvaluatedResponseFunction();

    //! Destructor
    virtual ~EvaluatedResponseFunction();

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


    //! Post process responses
    virtual void 
    postProcessResponses(const Epetra_Comm& comm, Teuchos::RCP<Epetra_Vector>& g);

    //! Post process response derivatives
    virtual void 
    postProcessResponseDerivatives(const Epetra_Comm& comm, Teuchos::RCP<Epetra_MultiVector>& gt);


    //! Set initial values (and number) of the responses.  This function is called by
    //   response evaluators, which act on a single workset at a time.
    void setResponseInitialValues(const std::vector<double>& initVals);
    void setResponseInitialValues(double singleInitValForAll, unsigned int numberOfResponses);

    //! Set post processing parameter list
    void setPostProcessingParams(const Teuchos::ParameterList& params);


  private:

    //! Private to prohibit copying
    EvaluatedResponseFunction(const EvaluatedResponseFunction&);
    
    //! Private to prohibit copying
    EvaluatedResponseFunction& operator=(const EvaluatedResponseFunction&);

    //! initial values for each response.  The length of this
    //  vector determines the number of responses.
    std::vector<double> responseInitVals;

    //! post processing parameter list
    Teuchos::ParameterList postProcessingParams;

  };

}

#endif // ALBANY_EVALUATEDRESPONSEFUNCTION_HPP
