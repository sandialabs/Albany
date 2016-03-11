//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PeridigmManager.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "AlbanyPeridigmOBCFunctional.hpp"

Albany::AlbanyPeridigmOBCFunctional::
AlbanyPeridigmOBCFunctional(const Teuchos::RCP<const Teuchos_Comm>& commT) :
			    Albany::ScalarResponseFunction(commT)
{}

Albany::AlbanyPeridigmOBCFunctional::
~AlbanyPeridigmOBCFunctional()
{
}

unsigned int
Albany::AlbanyPeridigmOBCFunctional::
numResponses() const
{
  return 1;
}


// **********************************************************************

void Albany::AlbanyPeridigmOBCFunctional::
evaluateResponseT(const double current_time,
     const Tpetra_Vector* xdotT,
     const Tpetra_Vector* xdotdotT,
     const Tpetra_Vector& xT,
     const Teuchos::Array<ParamVec>& p,
     Tpetra_Vector& gT)
{
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst();
  LCM::PeridigmManager& peridigmManager = *LCM::PeridigmManager::self();
  gT_nonconstView[0] = peridigmManager.obcEvaluateFunctional();
}

void
Albany::AlbanyPeridigmOBCFunctional::
evaluateTangentT(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
		const Tpetra_Vector* xdotT,
		const Tpetra_Vector* xdotdotT,
		const Tpetra_Vector& xT,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Tpetra_MultiVector* VxdotT,
		const Tpetra_MultiVector* VxdotdotT,
		const Tpetra_MultiVector* VxT,
		const Tpetra_MultiVector* VpT,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{}


#ifdef ALBANY_EPETRA
void
Albany::AlbanyPeridigmOBCFunctional::
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
     Epetra_MultiVector* dg_dp)
{

  // Evaluate response g
  if ((g != NULL) || dg_dx != NULL) {
    LCM::PeridigmManager& peridigmManager = *LCM::PeridigmManager::self();
    Epetra_Vector* dgdx0 = (dg_dx != NULL) ? (*dg_dx)(0) : NULL;
    double resp = peridigmManager.obcEvaluateFunctional((*dg_dx)(0));
    if (g != NULL)
      (*g)[0] = resp;
  }

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);
  if (dg_dxdotdot != NULL)
    dg_dxdotdot->PutScalar(0.0);

  // Evaluate dg/dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}
#endif

//! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
void
Albany::AlbanyPeridigmOBCFunctional::
evaluateGradientT(const double current_time,
     const Tpetra_Vector* xdotT,
     const Tpetra_Vector* xdotdotT,
     const Tpetra_Vector& xT,
     const Teuchos::Array<ParamVec>& p,
     ParamVec* deriv_p,
     Tpetra_Vector* gT,
     Tpetra_MultiVector* dg_dxT,
     Tpetra_MultiVector* dg_dxdotT,
     Tpetra_MultiVector* dg_dxdotdotT,
     Tpetra_MultiVector* dg_dpT){};

#ifdef ALBANY_EPETRA
//! Evaluate distributed parameter derivative dg/dp
void
Albany::AlbanyPeridigmOBCFunctional::
evaluateDistParamDeriv(
         const double current_time,
         const Epetra_Vector* xdot,
         const Epetra_Vector* xdotdot,
         const Epetra_Vector& x,
         const Teuchos::Array<ParamVec>& param_array,
         const std::string& dist_param_name,
         Epetra_MultiVector* dg_dp){
  dg_dp->PutScalar(0.0);
};
#endif

#ifdef ALBANY_SG
void
Albany::AlbanyPeridigmOBCFunctional::
evaluateSGResponse(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType>>& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g)
{}

void
Albany::AlbanyPeridigmOBCFunctional::
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
  const Teuchos::Array< Teuchos::Array<SGType>>& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp)
{}

void
Albany::AlbanyPeridigmOBCFunctional::
evaluateSGGradient(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType>>& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp)
{}
#endif 
#ifdef ALBANY_ENSEMBLE 

void
Albany::AlbanyPeridigmOBCFunctional::
evaluateMPResponse(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType>>& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g)
{}

void
Albany::AlbanyPeridigmOBCFunctional::
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
  const Teuchos::Array< Teuchos::Array<MPType>>& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_JV,
  Stokhos::ProductEpetraMultiVector* mp_gp)
{}

void
Albany::AlbanyPeridigmOBCFunctional::
evaluateMPGradient(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType>>& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_dg_dx,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dp)
{}
#endif
