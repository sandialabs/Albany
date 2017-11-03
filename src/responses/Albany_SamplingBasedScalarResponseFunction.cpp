//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#ifdef ALBANY_EPETRA
#include "Petra_Converters.hpp"
#endif

using Teuchos::RCP;
using Teuchos::rcp;

namespace {
#ifdef ALBANY_EPETRA
class SGConverter : public Petra::Converter {
public:
  SGConverter (Albany::AbstractResponseFunction* arf,
               const Teuchos::RCP<const Teuchos_Comm>& commT)
    : arf_(arf),
      Petra::Converter(commT)
  {}

  void evaluateResponse (
    const double current_time,
    const Epetra_Vector* xdot,
    const Epetra_Vector* xdotdot,
    const Epetra_Vector& x,
    const Teuchos::Array<ParamVec>& p,
    Epetra_Vector& g)
  {
    RCP<Tpetra_Vector> gT = e2t(g);
    arf_->evaluateResponseT(
      current_time, e2t(xdot).get(), e2t(xdotdot).get(), *e2t(x), p, *gT);
    t2e(gT, g);
  }

  void evaluateTangent (
    const double alpha,
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
    Epetra_MultiVector* gp)
  {
    RCP<Tpetra_Vector> gT = e2t(g);
    RCP<Tpetra_MultiVector> gxT = e2t(gx), gpT = e2t(gp);
    arf_->evaluateTangentT(
      alpha, beta, omega, current_time, sum_derivs, e2t(xdot).get(),
      e2t(xdotdot).get(), *e2t(x), p, deriv_p, e2t(Vxdot).get(),
      e2t(Vxdotdot).get(), e2t(Vx).get(), e2t(Vp).get(), gT.get(), gxT.get(),
      gpT.get());
    t2e(gT, g);
    t2e(gxT, gx);
    t2e(gpT, gp);
  }

private:
  Albany::AbstractResponseFunction* arf_;
};
#endif
}

