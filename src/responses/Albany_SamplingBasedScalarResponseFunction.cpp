//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

#ifdef ALBANY_SG
void
Albany::SamplingBasedScalarResponseFunction::
init_sg(
  const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis_,
  const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad_,
  const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion_,
  const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm_)
{
  basis = basis_;
  quad = quad_;
}

void
Albany::SamplingBasedScalarResponseFunction::
evaluateSGResponse(
  const double curr_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g)
{
  RCP<const Epetra_BlockMap> x_map = sg_x.coefficientMap();
  RCP<Epetra_Vector> xdot;
  if (sg_xdot != NULL)
    xdot = rcp(new Epetra_Vector(*x_map));
  RCP<Epetra_Vector> xdotdot;
  if (sg_xdotdot != NULL)
    xdotdot = rcp(new Epetra_Vector(*x_map));
  Epetra_Vector x(*x_map);
  Teuchos::Array<ParamVec> pp = p;
  
  RCP<const Epetra_BlockMap> g_map = sg_g.coefficientMap();
  Epetra_Vector g(*g_map);

  // Get quadrature data
  const Teuchos::Array<double>& norms = basis->norm_squared();
  const Teuchos::Array< Teuchos::Array<double> >& points = 
    quad->getQuadPoints();
  const Teuchos::Array<double>& weights = quad->getQuadWeights();
  const Teuchos::Array< Teuchos::Array<double> >& vals = 
    quad->getBasisAtQuadPoints();
  int nqp = points.size();

  // Compute sg_g via quadrature
  sg_g.init(0.0);
  SGConverter c(this, commT);
  for (int qp=0; qp<nqp; qp++) {

    // Evaluate sg_x, sg_xdot at quadrature point
    sg_x.evaluate(vals[qp], x);
    if (sg_xdot != NULL)
      sg_xdot->evaluate(vals[qp], *xdot);
    if (sg_xdotdot != NULL)
      sg_xdotdot->evaluate(vals[qp], *xdotdot);

    // Evaluate parameters at quadrature point
    for (int i=0; i<sg_p_index.size(); i++) {
      int ii = sg_p_index[i];
      for (unsigned int j=0; j<pp[ii].size(); j++)
	pp[ii][j].baseValue = sg_p_vals[ii][j].evaluate(points[qp], vals[qp]);
    }

    // Compute response at quadrature point
    c.evaluateResponse(curr_time, xdot.get(), xdotdot.get(), x, pp, g);

    // Add result into integral
    sg_g.sumIntoAllTerms(weights[qp], vals[qp], norms, g);
  }
}

void
Albany::SamplingBasedScalarResponseFunction::
evaluateSGTangent(
  const double alpha, 
  const double beta, 
  const double omega, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp)
{
  RCP<const Epetra_BlockMap> x_map = sg_x.coefficientMap();
  RCP<Epetra_Vector> xdot;
  if (sg_xdot != NULL)
    xdot = rcp(new Epetra_Vector(*x_map));
  RCP<Epetra_Vector> xdotdot;
  if (sg_xdotdot != NULL)
    xdotdot = rcp(new Epetra_Vector(*x_map));
  Epetra_Vector x(*x_map);
  Teuchos::Array<ParamVec> pp = p;
  
  RCP<Epetra_Vector> g;
  if (sg_g != NULL) {
    sg_g->init(0.0);
    g = rcp(new Epetra_Vector(*(sg_g->coefficientMap())));
  }

  RCP<Epetra_MultiVector> JV;
  if (sg_JV != NULL) {
    sg_JV->init(0.0);
    JV = rcp(new Epetra_MultiVector(*(sg_JV->coefficientMap()), 
				    sg_JV->numVectors()));
  }

  RCP<Epetra_MultiVector> gp;
  if (sg_gp != NULL) {
    sg_gp->init(0.0);
    gp = rcp(new Epetra_MultiVector(*(sg_gp->coefficientMap()), 
				    sg_gp->numVectors()));
  }

  // Get quadrature data
  const Teuchos::Array<double>& norms = basis->norm_squared();
  const Teuchos::Array< Teuchos::Array<double> >& points = 
    quad->getQuadPoints();
  const Teuchos::Array<double>& weights = quad->getQuadWeights();
  const Teuchos::Array< Teuchos::Array<double> >& vals = 
    quad->getBasisAtQuadPoints();
  int nqp = points.size();

  // Compute sg_g via quadrature
  SGConverter c(this, commT);
  for (int qp=0; qp<nqp; qp++) {

    // Evaluate sg_x, sg_xdot at quadrature point
    sg_x.evaluate(vals[qp], x);
    if (sg_xdot != NULL)
      sg_xdot->evaluate(vals[qp], *xdot);
    if (sg_xdotdot != NULL)
      sg_xdotdot->evaluate(vals[qp], *xdotdot);

    // Evaluate parameters at quadrature point
    for (int i=0; i<sg_p_index.size(); i++) {
      int ii = sg_p_index[i];
      for (unsigned int j=0; j<pp[ii].size(); j++) {
	pp[ii][j].baseValue = sg_p_vals[ii][j].evaluate(points[qp], vals[qp]);
	if (deriv_p != NULL) {
	  for (unsigned int k=0; k<deriv_p->size(); k++)
	    if ((*deriv_p)[k].family->getName() == pp[ii][j].family->getName())
	      (*deriv_p)[k].baseValue = pp[ii][j].baseValue;
	}
      }
    }

    // Compute response at quadrature point
    c.evaluateTangent(alpha, beta, omega, current_time, sum_derivs, 
                      xdot.get(), xdotdot.get(), x, pp, deriv_p, Vxdot, Vxdotdot, Vx, Vp,
                      g.get(), JV.get(), gp.get());
    
    // Add result into integral
    if (sg_g != NULL)
      sg_g->sumIntoAllTerms(weights[qp], vals[qp], norms, *g);
    if (sg_JV != NULL)
      sg_JV->sumIntoAllTerms(weights[qp], vals[qp], norms, *JV);
    if (sg_gp != NULL)
      sg_gp->sumIntoAllTerms(weights[qp], vals[qp], norms, *gp);
  }
}

void
Albany::SamplingBasedScalarResponseFunction::
evaluateSGGradient(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp)
{
  RCP<const Epetra_BlockMap> x_map = sg_x.coefficientMap();
  RCP<Epetra_Vector> xdot;
  if (sg_xdot != NULL)
    xdot = rcp(new Epetra_Vector(*x_map));
  RCP<Epetra_Vector> xdotdot;
  if (sg_xdotdot != NULL)
    xdotdot = rcp(new Epetra_Vector(*x_map));
  Epetra_Vector x(*x_map);
  Teuchos::Array<ParamVec> pp = p;

  RCP<Epetra_Vector> g;
  if (sg_g != NULL) {
    sg_g->init(0.0);
    g = rcp(new Epetra_Vector(*(sg_g->coefficientMap())));
  }

  RCP<Epetra_MultiVector> dg_dx;
  if (sg_dg_dx != NULL) {
    sg_dg_dx->init(0.0);
    dg_dx = rcp(new Epetra_MultiVector(*(sg_dg_dx->coefficientMap()), 
				       sg_dg_dx->numVectors()));
  }

  RCP<Epetra_MultiVector> dg_dxdot;
  if (sg_dg_dxdot != NULL) {
    sg_dg_dxdot->init(0.0);
    dg_dxdot = rcp(new Epetra_MultiVector(*(sg_dg_dxdot->coefficientMap()), 
					  sg_dg_dxdot->numVectors()));
  }

  RCP<Epetra_MultiVector> dg_dxdotdot;
  if (sg_dg_dxdotdot != NULL) {
    sg_dg_dxdotdot->init(0.0);
    dg_dxdotdot = rcp(new Epetra_MultiVector(*(sg_dg_dxdotdot->coefficientMap()), 
					  sg_dg_dxdotdot->numVectors()));
  }

  RCP<Epetra_MultiVector> dg_dp;
  if (sg_dg_dp != NULL) {
    sg_dg_dp->init(0.0);
    dg_dp = rcp(new Epetra_MultiVector(*(sg_dg_dp->coefficientMap()), 
				       sg_dg_dp->numVectors()));
  }

  // Get quadrature data
  const Teuchos::Array<double>& norms = basis->norm_squared();
  const Teuchos::Array< Teuchos::Array<double> >& points = 
    quad->getQuadPoints();
  const Teuchos::Array<double>& weights = quad->getQuadWeights();
  const Teuchos::Array< Teuchos::Array<double> >& vals = 
    quad->getBasisAtQuadPoints();
  int nqp = points.size();

  // Compute sg_g via quadrature
  for (int qp=0; qp<nqp; qp++) {

    // Evaluate sg_x, sg_xdot at quadrature point
    sg_x.evaluate(vals[qp], x);
    if (sg_xdot != NULL)
      sg_xdot->evaluate(vals[qp], *xdot);
    if (sg_xdotdot != NULL)
      sg_xdotdot->evaluate(vals[qp], *xdotdot);

    // Evaluate parameters at quadrature point
    for (int i=0; i<sg_p_index.size(); i++) {
      int ii = sg_p_index[i];
      for (unsigned int j=0; j<pp[ii].size(); j++) {
	pp[ii][j].baseValue = sg_p_vals[ii][j].evaluate(points[qp], vals[qp]);
	if (deriv_p != NULL) {
	  for (unsigned int k=0; k<deriv_p->size(); k++)
	    if ((*deriv_p)[k].family->getName() == pp[ii][j].family->getName())
	      (*deriv_p)[k].baseValue = pp[ii][j].baseValue;
	}
      }
    }

    // Compute response at quadrature point
    evaluateGradient(current_time, xdot.get(), xdotdot.get(), x, pp, deriv_p,
		     g.get(), dg_dx.get(), dg_dxdot.get(), dg_dxdotdot.get(), dg_dp.get());

    // Add result into integral
    if (sg_g != NULL)
      sg_g->sumIntoAllTerms(weights[qp], vals[qp], norms, *g);
    if (sg_dg_dx != NULL)
      sg_dg_dx->sumIntoAllTerms(weights[qp], vals[qp], norms, *dg_dx);
    if (sg_dg_dxdot != NULL)
      sg_dg_dxdot->sumIntoAllTerms(weights[qp], vals[qp], norms, *dg_dxdot);
    if (sg_dg_dxdotdot != NULL)
      sg_dg_dxdotdot->sumIntoAllTerms(weights[qp], vals[qp], norms, *dg_dxdotdot);
    if (sg_dg_dp != NULL)
      sg_dg_dp->sumIntoAllTerms(weights[qp], vals[qp], norms, *dg_dp);
  }
}
#endif 
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
