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

namespace Albany {

  //! Tpetra_Operator implementing the action of df/dp (transpose)
  /*!
   * This class implements the Tpetra_Operator interface for
   * op(df/dp)*v where op() is the identity or tranpose, f is the Albany
   * residual vector, p is a distributed parameter vector, and v is a given
   * vector.
   */
  class DistributedParameterDerivativeOpT : public Tpetra_Operator {
  public:

    // Constructor
    DistributedParameterDerivativeOpT(
      const Teuchos::RCP<Application>& app_,
      const std::string& param_name_) :
      app(app_),
      param_name(param_name_),
      use_transpose(false) {}

    //! Destructor
    virtual ~DistributedParameterDerivativeOpT() {}

    //! Set values needed for apply()
    void set(const double time_,
             const Teuchos::RCP<const Tpetra_Vector>& xdot_,
             const Teuchos::RCP<const Tpetra_Vector>& xdotdot_,
             const Teuchos::RCP<const Tpetra_Vector>& x_,
             const Teuchos::RCP<Teuchos::Array<ParamVec> >& scalar_params_) {
      time = time_;
      xdot = xdot_;
      xdotdot = xdotdot_;
      x = x_;
      scalar_params = scalar_params_;
    }

    //! @name Tpetra_Operator methods
    //@{

    //! If set true, transpose of this operator will be applied.
    virtual int SetUseTranspose(bool UseTranspose) {
      use_transpose = UseTranspose;
      return 0;
    }

    /*!
     * \brief Returns the result of a Tpetra_Operator applied to a
     * Tpetra_MultiVector X in Y.
     */
    virtual void apply(const Tpetra_MultiVector& X,
                      Tpetra_MultiVector& Y,  Teuchos::ETransp  mode = Teuchos::NO_TRANS, 
                      ST alpha = Teuchos::ScalarTraits<ST>::one(), 
                      ST beta = Teuchos::ScalarTraits<ST>::one() ) const {
      app->applyGlobalDistParamDerivImplT(time,
                                     xdot,
                                     xdotdot,
                                     x,
                                     *scalar_params,
                                     param_name,
                                     use_transpose,
                                     Teuchos::rcpFromRef(X),
                                     Teuchos::rcpFromRef(Y));
    }


    //! Returns a character string describing the operator
    virtual const char * Label() const {
      return "DistributedParameterDerivativeOpT";
    }

     virtual bool hasTransposeApply() const {
       return use_transpose; 
     }

    /*!
     * \brief Returns the Tpetra_Map object associated with the domain of
     * this operator.
     */
    //virtual const Tpetra_Map& OperatorDomainMap() const {
    virtual Teuchos::RCP<const Tpetra_Map> getDomainMap() const {
      if (use_transpose)
        return app->getMapT();
      return app->getDistParamLib()->get(param_name)->map();
    }

    /*!
     * \brief Returns the Tpetra_Map object associated with the range of
     * this operator.
     */
    //virtual const Tpetra_Map& OperatorRangeMap() const {
    virtual Teuchos::RCP<const Tpetra_Map> getRangeMap() const {
      if (use_transpose)
        return app->getDistParamLib()->get(param_name)->map();
      return app->getMapT();
    }

    //@}

  protected:

    //! Albany applications
    Teuchos::RCP<Application> app;

    //! Name of distributed parameter we are differentiating w.r.t.
    std::string param_name;

    //! Whether to apply transpose
    bool use_transpose;

    //! @name Data needed for apply()
    //@{

    //! Current time
    double time;

    //! Velocity vector
    Teuchos::RCP<const Tpetra_Vector> xdot;

    //! Acceleration vector
    Teuchos::RCP<const Tpetra_Vector> xdotdot;

    //! Solution vector
    Teuchos::RCP<const Tpetra_Vector> x;

    //! Scalar parameters
    Teuchos::RCP<Teuchos::Array<ParamVec> > scalar_params;

    //@}

  }; // class DistributedParameterDerivativeOp

} // namespace Albany

#endif // ALBANY_DISTRIBUTED_PARAMETER_DERIVATIVE_OP_HPP
