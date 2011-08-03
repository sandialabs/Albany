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


#ifndef ALBANY_SOLUTIONMAXVALUERESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONMAXVALUERESPONSEFUNCTION_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include "Epetra_Vector.h"

namespace Albany {

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class SolutionMaxValueResponseFunction : public AbstractResponseFunction {
  public:
  
    //! Default constructor
    SolutionMaxValueResponseFunction(int neq = 1, int eq = 0, bool interleavedOrdering=true);

    //! Destructor
    virtual ~SolutionMaxValueResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void 
    evaluateResponses(const Epetra_Vector* xdot,
                      const Epetra_Vector& x,
                      const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
                      Epetra_Vector& g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangents(
	   const Epetra_Vector* xdot,
	   const Epetra_Vector& x,
	   const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	   const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dxdot_dp,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dx_dp,
	   Epetra_Vector* g,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& gt);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void 
    evaluateGradients(
	  const Epetra_Vector* xdot,
	  const Epetra_Vector& x,
	  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	  const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	  Epetra_Vector* g,
	  Epetra_MultiVector* dg_dx,
	  Epetra_MultiVector* dg_dxdot,
	  const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dg_dp);

    //! Evaluate stochastic Galerkin responses
    virtual void 
    evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
			const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
			const ParamVec* p,
			const ParamVec* sg_p,
			const Teuchos::Array<SGType>* sg_p_vals,
			Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g);

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
