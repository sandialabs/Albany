//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP
#define ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP

#include "Albany_ScalarResponseFunction.hpp"

namespace Albany {
/**
 * \brief Response Description
 */
class AlbanyPeridigmOBCFunctional : public ScalarResponseFunction
{
 public:
  AlbanyPeridigmOBCFunctional(const Teuchos::RCP<const Teuchos_Comm>& commT);

  //! Destructor
  virtual ~AlbanyPeridigmOBCFunctional();

  //! Get the number of responses
  virtual unsigned int
  numResponses() const;

  //! Evaluate responses
  virtual void
  evaluateResponse(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      Tpetra_Vector&                          gT);

  //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  virtual void
  evaluateTangent(
      const double                                 alpha,
      const double                                 beta,
      const double                                 omega,
      const double                                 current_time,
      bool                                         sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>&      x,
      const Teuchos::RCP<const Thyra_Vector>&      xdot,
      const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
      const Teuchos::Array<ParamVec>&              p,
      ParamVec*                                    deriv_p,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      Tpetra_Vector*                               g,
      Tpetra_MultiVector*                          gx,
      Tpetra_MultiVector*                          gp);

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  virtual void
  evaluateGradient(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      ParamVec*                               deriv_p,
      Tpetra_Vector*                          gT,
      Tpetra_MultiVector*                     dg_dxT,
      Tpetra_MultiVector*                     dg_dxdotT,
      Tpetra_MultiVector*                     dg_dxdotdotT,
      Tpetra_MultiVector*                     dg_dpT);

  //! Evaluate distributed parameter derivative dg/dp
  virtual void
  evaluateDistParamDeriv(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      dist_param_name,
      Tpetra_MultiVector*                     dg_dpT);

 private:
  //! Private to prohibit copying
  AlbanyPeridigmOBCFunctional(const AlbanyPeridigmOBCFunctional&);

  //! Private to prohibit copying
  AlbanyPeridigmOBCFunctional&
  operator=(const AlbanyPeridigmOBCFunctional&);
};

}  // namespace Albany

#endif  // ALBANY_PERIDIGM_OBC_FUNCTIONAL_HPP
