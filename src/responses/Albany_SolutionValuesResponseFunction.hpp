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
#if defined(ALBANY_EPETRA)
#include "Epetra_CombineMode.h"
#endif

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

#if defined(ALBANY_EPETRA)
    //! Setup response function
    virtual void setup();
#endif

    //! Setup response function
    virtual void setupT();

#if defined(ALBANY_EPETRA)
    //! Evaluate responses
    virtual void
    evaluateResponse(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector* xdotdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void
    evaluateTangent(const double alpha,
		    const double beta,
		    const double omega,
		    const double current_time,
		    bool sum_derivs,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector* xdotdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    ParamVec* deriv_p,
		    const Epetra_MultiVector* Vxdot,
		    const Epetra_MultiVector* Vxdotdot,
		    const Epetra_MultiVector* Vx,
		    const Epetra_MultiVector* Vp,
		    Epetra_Vector* g,
		    Epetra_MultiVector* gx,
		    Epetra_MultiVector* gp);

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
#endif // ALBANY_EPETRA

  private:
    //! Private to prohibit copying
    SolutionValuesResponseFunction(const SolutionValuesResponseFunction&);

    //! Private to prohibit copying
    SolutionValuesResponseFunction& operator=(const SolutionValuesResponseFunction&);

    Teuchos::RCP<const Application> app_;

    Teuchos::RCP<SolutionCullingStrategyBase> cullingStrategy_;
#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_Import> solutionImporter_;
#endif
    Teuchos::RCP<Tpetra_Import> solutionImporterT_;

    class SolutionPrinter;
    Teuchos::RCP<SolutionPrinter> sol_printer_;

#if defined(ALBANY_EPETRA)
    void updateSolutionImporter();
#endif
    void updateSolutionImporterT();

#if defined(ALBANY_EPETRA)
void
ImportWithAlternateMap(
    const Epetra_Import &importer,
    const Epetra_MultiVector &source,
    Epetra_MultiVector &target,
    Epetra_CombineMode mode);
#endif

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
