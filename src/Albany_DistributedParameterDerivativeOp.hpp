//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP

#include "Albany_Application.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_DistributedParameterLibrary.hpp"

#include "Thyra_MultiVectorStdOps.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

  //! Thyra_LinearOp implementing the action of df/dp (transpose)
  /*!
   * This class implements the Thyra::LinearOpBase interface for
   * op(df/dp)*v where op() is the identity or tranpose, f is the Albany
   * residual vector, p is a distributed parameter vector, and v is a given
   * vector.
   */
  class DistributedParameterDerivativeOp : public Thyra_LinearOp {
  public:

    // Constructor
    DistributedParameterDerivativeOp(
      const Teuchos::RCP<Application>& app_,
      const std::string& param_name_) :
      app(app_),
      param_name(param_name_) {}

    //! Destructor
    virtual ~DistributedParameterDerivativeOp() {}

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
      return app->getDistributedParameterLibrary()->get(param_name)->vector_space();
    }

    //! Overrides Thyra::LinearOpBase purely virtual method
    Teuchos::RCP<const Thyra_VectorSpace> range() const {
      return app->getVectorSpace();
    }

    //@}

  protected:
    //! Overrides Thyra::LinearOpBase purely virtual method
    bool opSupportedImpl(Thyra::EOpTransp /*M_trans*/) const {
      // The underlying scalar type is not complex, and we support transpose, so we support everything.
      return true;
    }

    //! Overrides Thyra::LinearOpBase purely virtual method
    //! Y = alpha*op(df/dp)*X + beta*Y
    void applyImpl (const Thyra::EOpTransp M_trans,
                    const Thyra_MultiVector& X,
                    const Teuchos::Ptr<Thyra_MultiVector>& Y,
                    const ST alpha,
                    const ST beta) const {

      bool use_transpose = (Thyra::real_trans(M_trans) == Thyra::TRANS);

      // applyGlobalDistParamDerivImpl overwrites its output: unless Y = op(df/dp)*X,
      // compute op(df/dp)*X in a work multivector, then update Y.
      const bool overwrite = (alpha == ST(1.0)) && (beta == ST(0.0));
      Teuchos::RCP<Thyra_MultiVector> opX = overwrite ? Teuchos::rcpFromPtr(Y) :
          Thyra::createMembers(Y->range(), Y->domain()->dim());
      app->applyGlobalDistParamDerivImpl(time, x, xdot, xdotdot,
                                         *scalar_params,
                                         param_name,
                                         use_transpose,
                                         Teuchos::rcpFromRef(X),
                                         opX);
      if (!overwrite) {
        if (beta == ST(0.0)) {
          Y->assign(0.0);
        } else if (beta != ST(1.0)) {
          Thyra::scale(beta, Y);
        }
        Thyra::update(alpha, *opX, Y);
      }
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

#endif // ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP
