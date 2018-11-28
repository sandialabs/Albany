//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_T_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_T_HPP

#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp" 

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_Application.hpp"

#include "Albany_TpetraThyraUtils.hpp"

namespace Albany {

  //! Tpetra_Operator implementing the action of df/dp (transpose)
  /*!
   * This class implements the Tpetra_Operator interface for
   * op(df/dp)*v where op() is the identity or tranpose, f is the Albany
   * residual vector, p is a distributed parameter vector, and v is a given
   * vector.
   */
  class DistributedParameterDerivativeOpT : public Thyra_LinearOp {
  public:

    // Constructor
    DistributedParameterDerivativeOpT(
      const Teuchos::RCP<Application>& app_,
      const std::string& param_name_) :
      app(app_),
      param_name(param_name_) {}

    //! Destructor
    virtual ~DistributedParameterDerivativeOpT() {}

    //! Set values needed for apply()
    void set(const double time_,
             const Teuchos::RCP<const Thyra_Vector>& x_,
             const Teuchos::RCP<const Thyra_Vector>& xdot_,
             const Teuchos::RCP<const Thyra_Vector>& xdotdot_,
             const Teuchos::RCP<Teuchos::Array<ParamVec> >& scalar_params_) {
      time = time_;
      xdot = xdot_;
      xdotdot = xdotdot_;
      x = x_;
      scalar_params = scalar_params_;
    }

    //! Overrides Thyra::LinearOpBase purely virtual method
    Teuchos::RCP<const Thyra_VectorSpace> domain() const {
      return Thyra::createVectorSpace<ST>(app->getDistParamLib()->get(param_name)->map());
    }

    //! Overrides Thyra::LinearOpBase purely virtual method
    Teuchos::RCP<const Thyra_VectorSpace> range() const {
      return Thyra::createVectorSpace<ST>(app->getMapT());
    }

    //@}

  protected:
    //! Overrides Thyra::LinearOpBase purely virtual method
    bool opSupportedImpl(Thyra::EOpTransp /*M_trans*/) const {
      // The underlying scalar type is not complex, and we support transpose, so we support everything.
      return true;
    }

    //! Overrides Thyra::LinearOpBase purely virtual method
    void applyImpl (const Thyra::EOpTransp M_trans,
                    const Thyra_MultiVector& X,
                    const Teuchos::Ptr<Thyra_MultiVector>& Y,
                    const ST alpha,
                    const ST beta) const {

      bool use_transpose = (M_trans == Thyra::TRANS);
      app->applyGlobalDistParamDerivImpl(time, x, xdot, xdotdot,
                                         *scalar_params,
                                         param_name,
                                         use_transpose,
                                         Teuchos::rcpFromRef(X),
                                         Teuchos::rcpFromPtr(Y));
    }

    //! Albany applications
    Teuchos::RCP<Application> app;

    //! Name of distributed parameter we are differentiating w.r.t.
    std::string param_name;

    //! @name Data needed for apply()
    //@{

    //! Current time
    double time;

    //! Velocity vector
    Teuchos::RCP<const Thyra_Vector> xdot;

    //! Acceleration vector
    Teuchos::RCP<const Thyra_Vector> xdotdot;

    //! Solution vector
    Teuchos::RCP<const Thyra_Vector> x;

    //! Scalar parameters
    Teuchos::RCP<Teuchos::Array<ParamVec> > scalar_params;

    //@}

  }; // class DistributedParameterDerivativeOp

} // namespace Albany

#endif // ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_T_HPP
