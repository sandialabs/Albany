//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_SOLUTIONFILERESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONFILERESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

namespace Albany {

  /*!
   * \brief Response function representing the difference from a stored vector on disk
   */
  template<class VectorNorm>
  class SolutionFileResponseFunction : 
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    SolutionFileResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT);

    //! Destructor
    virtual ~SolutionFileResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Perform optimization setup
    virtual void postRegSetup(){};

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

    //! Evaluate distributed parameter derivative dg/dp
    virtual void
    evaluateDistParamDerivT(
        const double current_time,
        const Tpetra_Vector* xdotT,
        const Tpetra_Vector* xdotdotT,
        const Tpetra_Vector& xT,
        const Teuchos::Array<ParamVec>& param_array,
        const std::string& dist_param_name,
        Tpetra_MultiVector* dg_dpT);

  private:

    //! Private to prohibit copying
    SolutionFileResponseFunction(const SolutionFileResponseFunction&);
    
    //! Private to prohibit copying
    SolutionFileResponseFunction& operator=(const SolutionFileResponseFunction&);

    //! Reference Vector - Tpetra
    Tpetra_Vector* RefSolnT;

    bool solutionLoaded;


    int MatrixMarketFileToTpetraVector( const char *filename, const Tpetra_Map & map, Tpetra_Vector * & A);
    int MatrixMarketFileToTpetraMultiVector( const char *filename, const Tpetra_Map & map, Tpetra_MultiVector * & A);

  };

//	namespace SolutionFileResponseFunction {
	
	  struct NormTwo {
	    double NormT(const Tpetra_Vector& vecT){ Teuchos::ScalarTraits<ST>::magnitudeType normT = vecT.norm2(); return normT * normT;}
	    void NormDerivativeT(const Tpetra_Vector& xT, const Tpetra_Vector& solnT, Tpetra_MultiVector& gradT) {
	      gradT.update(2.0,xT,-2.0,solnT,0.0);
	    }
	  };
	
	  struct NormInf {
	    double NormT(const Tpetra_Vector& vecT){ Teuchos::ScalarTraits<ST>::magnitudeType normT = vecT.normInf(); return normT;}
	    void NormDerivativeT(const Tpetra_Vector& xT, const Tpetra_Vector& solnT, Tpetra_MultiVector& gradT) {
	      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
	        std::endl << "SolutionFileResponseFunction::NormInf::NormDerivativeT is not Implemented yet!"
	        << std::endl);
      }
	
	  };
//	}

}

// Define macro for explicit template instantiation
#define SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMTWO(name) \
  template class name<Albany::NormTwo>;
#define SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMINF(name) \
  template class name<Albany::NormInf>;

#define SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS(name) \
  SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMTWO(name) \
  SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMINF(name)

#endif // ALBANY_SOLUTIONFILERESPONSEFUNCTION_HPP
