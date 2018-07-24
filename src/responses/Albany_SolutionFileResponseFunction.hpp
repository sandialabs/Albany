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
    SolutionFileResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& comm);

    //! Destructor
    virtual ~SolutionFileResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Perform optimization setup
    virtual void postRegSetup(){};

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  Tpetra_Vector& gT);

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
		  Tpetra_Vector* g,
		  Tpetra_MultiVector* gx,
		  Tpetra_MultiVector* gp);
  
    virtual void 
    evaluateGradient(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  ParamVec* deriv_p,
		  Tpetra_Vector* gT,
		  Tpetra_MultiVector* dg_dxT,
		  Tpetra_MultiVector* dg_dxdotT,
		  Tpetra_MultiVector* dg_dxdotdotT,
		  Tpetra_MultiVector* dg_dpT);

    //! Evaluate distributed parameter derivative dg/dp
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Tpetra_MultiVector* dg_dpT);

  private:

    //! Private to prohibit copying
    SolutionFileResponseFunction(const SolutionFileResponseFunction&);
    
    //! Private to prohibit copying
    SolutionFileResponseFunction& operator=(const SolutionFileResponseFunction&);

    //! Reference Vector - Thyra
    Teuchos::RCP<Thyra_Vector> RefSoln;

    // A temp vector used in the response. Store it, so we create/destroy it only once.
    Teuchos::RCP<Thyra_Vector> diff;

    bool solutionLoaded;


    int MatrixMarketFileToThyraVector( const char *filename, const Teuchos::RCP<Thyra_Vector>& v);
    int MatrixMarketFileToTpetraMultiVector( const char *filename, Tpetra_MultiVector& AT);

  };

  struct NormTwo {
    static double Norm (const Thyra_Vector& vec)
    {
      auto norm = vec.norm_2();
      return norm * norm;
    }

    static void NormDerivative(const Thyra_Vector& x, const Thyra_Vector& soln, Thyra_Vector& grad)
    {
      Teuchos::Array<ST> coeffs(2);
      coeffs[0] = 2.0; coeffs[1] = -2.0;
      Teuchos::Array<Teuchos::Ptr<const Thyra_Vector>> vecs(2);
      vecs[0] = Teuchos::constPtr(x);
      vecs[1] = Teuchos::constPtr(soln);
      grad.linear_combination(coeffs,vecs,0.0);
    }
  };

  struct NormInf {
    static double Norm (const Thyra_Vector& vec)
    {
      return vec.norm_inf();
    }

    static void NormDerivative (const Thyra_Vector& x, const Thyra_Vector& soln, Thyra_Vector& grad)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "SolutionFileResponseFunction::NormInf::NormDerivativeT is not Implemented yet!\n");
    }
  };

} // namespace Albany

// Define macro for explicit template instantiation
#define SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMTWO(name) \
  template class name<Albany::NormTwo>;
#define SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMINF(name) \
  template class name<Albany::NormInf>;

#define SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS(name)    \
  SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMTWO(name)  \
  SOLFILERESP_INSTANTIATE_TEMPLATE_CLASS_NORMINF(name)

#endif // ALBANY_SOLUTIONFILERESPONSEFUNCTION_HPP
