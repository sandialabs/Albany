//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP

#include "Albany_DistributedResponseFunction.hpp"
#include "Albany_Application.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

  /*!
   * \brief A response function given by (possibly a portion of) the solution
   */
  class SolutionResponseFunction : public DistributedResponseFunction {
  public:
  
    //! Default constructor
    SolutionResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      Teuchos::ParameterList& responseParams);

    //! Destructor
    virtual ~SolutionResponseFunction();

    //! Setup response function
    virtual void setup();
    
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const;

    //! Create operator for gradient
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses
    virtual void evaluateResponse(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      Tpetra_Vector& gT);
    
    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void evaluateTangent(
      const double alpha, 
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

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp - Tpetra
    virtual void evaluateGradient(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* gT,
      Tpetra_Operator* dg_dxT,
      Tpetra_Operator* dg_dxdotT,
      Tpetra_Operator* dg_dxdotdotT,
      Tpetra_MultiVector* dg_dpT);

    //! Evaluate distributed parameter derivative = dg/dp
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Tpetra_MultiVector* dg_dpT);
    //@}

  private:

    //! Private to prohibit copying
    SolutionResponseFunction(const SolutionResponseFunction&);
    
    //! Private to prohibit copying
    SolutionResponseFunction& operator=(const SolutionResponseFunction&);

  protected:
    
    Teuchos::RCP<const Tpetra_Map> 
    buildCulledMapT(const Tpetra_Map& x_mapT, 
		   const Teuchos::Array<int>& keepDOF) const;
    
    //Tpetra version of above function
    void cullSolution(
        const Teuchos::RCP<const Thyra_MultiVector>& x, 
		    Tpetra_MultiVector& x_culledT) const;

  protected:

    //! Application to get global maps
    Teuchos::RCP<Albany::Application> application;

    //! Mask for DOFs to keep
    Teuchos::Array<int> keepDOF;

    
    //! Tpetra map for response
    Teuchos::RCP<const Tpetra_Map> culled_mapT;


    //! Tpetra importer mapping between full and culled solution
    Teuchos::RCP<Tpetra_Import> importerT;


    //! Graph of gradient operator - Tpetra version
    Teuchos::RCP<Tpetra_CrsGraph> gradient_graphT;

  };

} // namespace Albany

#endif // ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
