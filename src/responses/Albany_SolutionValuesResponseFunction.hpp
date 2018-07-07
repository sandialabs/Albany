//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTIONVALUESRESPONSEFUNCTION_HPP
#define ALBANY_SOLUTIONVALUESRESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"

#include "Albany_Application.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Albany_Utils.hpp"

#include "Tpetra_CombineMode.hpp"

namespace Albany {

  class SolutionCullingStrategyBase;

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class SolutionValuesResponseFunction :
    public SamplingBasedScalarResponseFunction {
  public:

    //! Constructor
    SolutionValuesResponseFunction(
      const Teuchos::RCP<const Application>& app,
      Teuchos::ParameterList& responseParams);

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Setup response function
    virtual void setupT();

    //! Evaluate responses
    virtual void
    evaluateResponseT(const double current_time,
		     const Tpetra_Vector* xdot,
		     const Tpetra_Vector* xdotdot,
		     const Tpetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Tpetra_Vector& g);

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

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void
    evaluateGradientT(const double current_time,
		     const Tpetra_Vector* xdot,
		     const Tpetra_Vector* xdotdot,
		     const Tpetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Tpetra_Vector* g,
		     Tpetra_MultiVector* dg_dx,
		     Tpetra_MultiVector* dg_dxdot,
		     Tpetra_MultiVector* dg_dxdotdot,
		     Tpetra_MultiVector* dg_dp);

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
    SolutionValuesResponseFunction(const SolutionValuesResponseFunction&);

    //! Private to prohibit copying
    SolutionValuesResponseFunction& operator=(const SolutionValuesResponseFunction&);

    Teuchos::RCP<const Application> app_;

    Teuchos::RCP<SolutionCullingStrategyBase> cullingStrategy_;

    Teuchos::RCP<Tpetra_Import> solutionImporterT_;

    class SolutionPrinter;
    Teuchos::RCP<SolutionPrinter> sol_printer_;

    void updateSolutionImporterT();

void
ImportWithAlternateMapT(
    Teuchos::RCP<const Tpetra_Import> importerT,
    const Tpetra_MultiVector& sourceT,
    Tpetra_MultiVector* targetT,
    Tpetra::CombineMode modeT);

void
ImportWithAlternateMapT(
    Teuchos::RCP<const Tpetra_Import> importerT,
    const Tpetra_Vector& sourceT,
    Tpetra_Vector& targetT,
    Tpetra::CombineMode modeT);

  };

}

#endif // ALBANY_SOLUTIONVALUESRESPONSEFUNCTION_HPP
