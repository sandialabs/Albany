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
    virtual void setupT();
    
    //! Get the map associate with this response
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const;

    //! Create operator for gradient
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses
    virtual void evaluateResponseT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& p,
      Tpetra_Vector& gT);
    
    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void evaluateTangentT(
      const double alpha, 
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

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp - Tpetra
    virtual void evaluateGradientT(
      const double current_time,
      const Tpetra_Vector* xdotT,
      const Tpetra_Vector* xdotdotT,
      const Tpetra_Vector& xT,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* gT,
      Tpetra_Operator* dg_dxT,
      Tpetra_Operator* dg_dxdotT,
      Tpetra_Operator* dg_dxdotdotT,
      Tpetra_MultiVector* dg_dpT);

    //! Evaluate distributed parameter derivative = dg/dp
    virtual void
    evaluateDistParamDerivT(
          const double current_time,
          const Tpetra_Vector* xdotT,
          const Tpetra_Vector* xdotdotT,
          const Tpetra_Vector& xT,
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
    void cullSolutionT(const Tpetra_MultiVector& xT, 
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

}

#endif // ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
