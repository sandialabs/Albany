//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACTRESPONSEFUNCTION_HPP
#define ALBANY_ABSTRACTRESPONSEFUNCTION_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Thyra_ModelEvaluatorBase.hpp"
#include "Albany_DataTypes.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a response function
   */
  class AbstractResponseFunction {
  public:

    //! Default constructor
    AbstractResponseFunction() {};

    //! Destructor
    virtual ~AbstractResponseFunction() {};

    //! Setup response function
    virtual void setup() = 0;

    //! Get the map associate with this response - Tpetra version 
    virtual Teuchos::RCP<const Tpetra_Map> responseMapT() const = 0;

    /*!
     * \brief Is this response function "scalar" valued, i.e., has a replicated
     * local response map.
     */
    virtual bool isScalarResponse() const = 0;

    //! Create Tpetra operator for gradient (e.g., dg/dx)
    virtual Teuchos::RCP<Tpetra_Operator> createGradientOpT() const = 0;

    //! perform post registration setup
    virtual void postRegSetup() = 0;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses
    virtual void evaluateResponse(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      Tpetra_Vector& gT) = 0;

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
      Tpetra_MultiVector* gp) = 0;

   virtual void evaluateDerivative(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Tpetra_Vector* gT,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
      const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp) = 0;

    //! Evaluate distributed parameter derivative dg/dp
    virtual void evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Tpetra_MultiVector*  dg_dpT) = 0;
    //@}

  private:

    //! Private to prohibit copying
    AbstractResponseFunction(const AbstractResponseFunction&);

    //! Private to prohibit copying
    AbstractResponseFunction& operator=(const AbstractResponseFunction&);

  };

} // namespace Albany

#endif // ALBANY_ABSTRACTRESPONSEFUNCTION_HPP
