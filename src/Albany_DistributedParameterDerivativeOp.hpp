//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP

#include "Epetra_Operator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_Application.hpp"

namespace Albany {

  //! Epetra_Operator implementing the action of df/dp (transpose)
  /*!
   * This class implements the Epetra_Operator interface for
   * op(df/dp)*v where op() is the identity or tranpose, f is the Albany
   * residual vector, p is a distributed parameter vector, and v is a given
   * vector.
   */
  class DistributedParameterDerivativeOp : public Epetra_Operator {
  public:

    // Constructor
    DistributedParameterDerivativeOp(
      const Teuchos::RCP<Application>& app_,
      const std::string& param_name_) :
      app(app_),
      param_name(param_name_),
      use_transpose(false) {}

    //! Destructor
    virtual ~DistributedParameterDerivativeOp() {}

    //! Set values needed for Apply()
    void set(const double time_,
             const Teuchos::RCP<const Epetra_Vector>& xdot_,
             const Teuchos::RCP<const Epetra_Vector>& xdotdot_,
             const Teuchos::RCP<const Epetra_Vector>& x_,
             const Teuchos::RCP<Teuchos::Array<ParamVec> >& scalar_params_) {
      time = time_;
      xdot = xdot_;
      xdotdot = xdotdot_;
      x = x_;
      scalar_params = scalar_params_;
    }

    //! @name Epetra_Operator methods
    //@{

    //! If set true, transpose of this operator will be applied.
    virtual int SetUseTranspose(bool UseTranspose) {
      use_transpose = UseTranspose;
    }

    /*!
     * \brief Returns the result of a Epetra_Operator applied to a
     * Epetra_MultiVector X in Y.
     */
    virtual int Apply(const Epetra_MultiVector& X,
                      Epetra_MultiVector& Y) const {
      app->applyGlobalDistParamDeriv(time,
                                     xdot.get(),
                                     xdotdot.get(),
                                     *x,
                                     *scalar_params,
                                     param_name,
                                     use_transpose,
                                     X,
                                     Y);
    }

    /*!
     * \brief Returns the result of a Epetra_Operator inverse applied to
     * an Epetra_MultiVector X in Y.
     */
    virtual int ApplyInverse(const Epetra_MultiVector& X,
                             Epetra_MultiVector& Y) const {
      TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Albany::DistributedParameterDerivativeOp does not support " <<
        "Epetra_Operator::ApplyInverse()!");
    }

    //! Returns the infinity norm of the global matrix.
    virtual double NormInf() const {
      return 0.0;
    }

    //! Returns a character string describing the operator
    virtual const char * Label() const {
      return "DistributedParameterDerivativeOp";
    }

    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const {
      return use_transpose;
    }

    /*!
     * \brief Returns true if the \e this object can provide an approximate
     * Inf-norm, false otherwise.
     */
    virtual bool HasNormInf() const {
      return false;
    }

    /*!
     * \brief Returns a pointer to the Epetra_Comm communicator associated
     * with this operator.
     */
    virtual const Epetra_Comm& Comm() const {
      return app->getDistParamLib()->get(param_name)->map()->Comm();
    }

    /*!
     * \brief Returns the Epetra_Map object associated with the domain of
     * this operator.
     */
    virtual const Epetra_Map& OperatorDomainMap() const {
      if (use_transpose)
        return *(app->getMap());
      return *(app->getDistParamLib()->get(param_name)->map());
    }

    /*!
     * \brief Returns the Epetra_Map object associated with the range of
     * this operator.
     */
    virtual const Epetra_Map& OperatorRangeMap() const {
      if (use_transpose)
        return *(app->getDistParamLib()->get(param_name)->map());
      return *(app->getMap());
    }

    //@}

  protected:

    //! Albany applications
    Teuchos::RCP<Application> app;

    //! Name of distributed parameter we are differentiating w.r.t.
    std::string param_name;

    //! Whether to apply transpose
    bool use_transpose;

    //! @name Data needed for Apply()
    //@{

    //! Current time
    double time;

    //! Velocity vector
    Teuchos::RCP<const Epetra_Vector> xdot;

    //! Acceleration vector
    Teuchos::RCP<const Epetra_Vector> xdotdot;

    //! Solution vector
    Teuchos::RCP<const Epetra_Vector> x;

    //! Scalar parameters
    Teuchos::RCP<Teuchos::Array<ParamVec> > scalar_params;

    //@}

  }; // class DistributedParameterDerivativeOp

} // namespace Albany

#endif // ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP
