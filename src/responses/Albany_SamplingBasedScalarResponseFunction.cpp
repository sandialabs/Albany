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

#ifdef ALBANY_ENSEMBLE 

void
Albany::SamplingBasedScalarResponseFunction::
evaluateMPResponse(
  const double curr_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g)
{
  Teuchos::Array<ParamVec> pp = p;
  const Epetra_Vector* xdot = NULL;
  const Epetra_Vector* xdotdot = NULL;

  SGConverter c(this, commT);
  for (int i=0; i<mp_x.size(); i++) {

    for (int k=0; k<mp_p_index.size(); k++) {
      int kk = mp_p_index[k];
      for (unsigned int j=0; j<pp[kk].size(); j++)
	pp[kk][j].baseValue = mp_p_vals[kk][j].coeff(i);
    }

    if (mp_xdot != NULL)
      xdot = mp_xdot->getCoeffPtr(i).get();
    if (mp_xdotdot != NULL)
      xdotdot = mp_xdotdot->getCoeffPtr(i).get();
    
    // Evaluate response function
    c.evaluateResponse(curr_time, xdot, xdotdot, mp_x[i], pp, mp_g[i]);
  }
}

void
Albany::SamplingBasedScalarResponseFunction::
evaluateMPTangent(
  const double alpha, 
  const double beta, 
  const double omega, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_JV,
  Stokhos::ProductEpetraMultiVector* mp_gp)
{
  Teuchos::Array<ParamVec> pp = p;
  const Epetra_Vector* xdot = NULL;
  const Epetra_Vector* xdotdot = NULL;
  Epetra_Vector* g = NULL;
  Epetra_MultiVector* JV = NULL;
  Epetra_MultiVector* gp = NULL;
  SGConverter c(this, commT);
  for (int i=0; i<mp_x.size(); i++) {

    for (int k=0; k<mp_p_index.size(); k++) {
      int kk = mp_p_index[k];
      for (unsigned int j=0; j<pp[kk].size(); j++) {
	pp[kk][j].baseValue = mp_p_vals[kk][j].coeff(i);
	if (deriv_p != NULL) {
	  for (unsigned int l=0; l<deriv_p->size(); l++)
	    if ((*deriv_p)[l].family->getName() == pp[kk][j].family->getName())
	      (*deriv_p)[l].baseValue = pp[kk][j].baseValue;
	}
      }
    }

    if (mp_xdot != NULL)
      xdot = mp_xdot->getCoeffPtr(i).get();
    if (mp_xdotdot != NULL)
      xdotdot = mp_xdotdot->getCoeffPtr(i).get();
    if (mp_g != NULL)
      g = mp_g->getCoeffPtr(i).get();
    if (mp_JV != NULL)
      JV = mp_JV->getCoeffPtr(i).get();
    if(mp_gp != NULL)
      gp = mp_gp->getCoeffPtr(i).get();
    
    // Evaluate response function
    c.evaluateTangent(alpha, beta, omega, current_time, sum_derivs,
                      xdot, xdotdot, mp_x[i], pp, deriv_p, Vxdot, Vxdotdot, Vx, Vp,
                      g, JV, gp);
  }
}

void
Albany::SamplingBasedScalarResponseFunction::
evaluateMPGradient(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_dg_dx,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dp)
{
  Teuchos::Array<ParamVec> pp = p;
  const Epetra_Vector* xdot = NULL;
  const Epetra_Vector* xdotdot = NULL;
  Epetra_Vector* g = NULL;
  Epetra_MultiVector* dg_dx = NULL;
  Epetra_MultiVector* dg_dxdot = NULL;
  Epetra_MultiVector* dg_dxdotdot = NULL;
  Epetra_MultiVector* dg_dp = NULL;
  for (int i=0; i<mp_x.size(); i++) {

    for (int k=0; k<mp_p_index.size(); k++) {
      int kk = mp_p_index[k];
      for (unsigned int j=0; j<pp[kk].size(); j++) {
	pp[kk][j].baseValue = mp_p_vals[kk][j].coeff(i);
	if (deriv_p != NULL) {
	  for (unsigned int l=0; l<deriv_p->size(); l++)
	    if ((*deriv_p)[l].family->getName() == pp[kk][j].family->getName())
	      (*deriv_p)[l].baseValue = pp[kk][j].baseValue;
	}
      }
    }

    if (mp_xdot != NULL)
      xdot = mp_xdot->getCoeffPtr(i).get();
    if (mp_xdotdot != NULL)
      xdotdot = mp_xdotdot->getCoeffPtr(i).get();
    if (mp_g != NULL)
      g = mp_g->getCoeffPtr(i).get();
    if (mp_dg_dx != NULL)
      dg_dx = mp_dg_dx->getCoeffPtr(i).get();
    if(mp_dg_dxdot != NULL)
      dg_dxdot = mp_dg_dxdot->getCoeffPtr(i).get();
    if(mp_dg_dxdotdot != NULL)
      dg_dxdotdot = mp_dg_dxdotdot->getCoeffPtr(i).get();
    if (mp_dg_dp != NULL)
      dg_dp = mp_dg_dp->getCoeffPtr(i).get();
    
    // Evaluate response function
    evaluateGradient(current_time, xdot, xdotdot,  mp_x[i], pp, deriv_p, 
		     g, dg_dx, dg_dxdot, dg_dxdotdot, dg_dp);
  }
}
#endif
