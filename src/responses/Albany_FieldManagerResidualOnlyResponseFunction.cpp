//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_FieldManagerResidualOnlyResponseFunction.hpp"

Albany::FieldManagerResidualOnlyResponseFunction::
FieldManagerResidualOnlyResponseFunction (
  const Teuchos::RCP<Albany::Application>& application_,
  const Teuchos::RCP<Albany::AbstractProblem>& problem_,
  const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<Albany::StateManager>& stateMgr_,
  Teuchos::ParameterList& responseParams) :
  FieldManagerScalarResponseFunction(application_, problem_, meshSpecs_, stateMgr_,
                                     responseParams)
{}

void Albany::FieldManagerResidualOnlyResponseFunction::
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
{
  // Evaluate just the response if it is requested.
  if (gT) this->evaluateResponseT(current_time, xdotT, xdotdotT, xT, p, *gT);
}

void Albany::FieldManagerResidualOnlyResponseFunction::
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
                  Tpetra_MultiVector* dg_dpT)
{
  if (gT) this->evaluateResponseT(current_time, xdotT, xdotdotT, xT, p, *gT);
}

#if defined(ALBANY_EPETRA)
void Albany::FieldManagerResidualOnlyResponseFunction:: 
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
  if (g)
    this->FieldManagerScalarResponseFunction::evaluateGradient(
      current_time, xdot, xdotdot, x, p, deriv_p, g, dg_dx, dg_dxdot,
      dg_dxdotdot, dg_dp);
}
#endif

#if defined(ALBANY_EPETRA)
void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateDistParamDeriv(
  const double current_time,
  const Epetra_Vector* xdot,
  const Epetra_Vector* xdotdot,
  const Epetra_Vector& x,
  const Teuchos::Array<ParamVec>& param_array,
  const std::string& dist_param_name,
  Epetra_MultiVector* dg_dp) {}
#endif

#ifdef ALBANY_SG
void Albany::FieldManagerResidualOnlyResponseFunction::
init_sg(
  const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
  const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
  const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
  const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm) {}

void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateSGResponse(
  const double curr_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g) {}

void Albany::FieldManagerResidualOnlyResponseFunction::
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
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp) {}

void Albany::FieldManagerResidualOnlyResponseFunction::
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
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp) {}
#endif

#ifdef ALBANY_ENSEMBLE 
void Albany::FieldManagerResidualOnlyResponseFunction::
evaluateMPResponse(
  const double curr_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g) {}

void Albany::FieldManagerResidualOnlyResponseFunction::
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
  Stokhos::ProductEpetraMultiVector* mp_gp) {}

void Albany::FieldManagerResidualOnlyResponseFunction::
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
  Stokhos::ProductEpetraMultiVector* mp_dg_dp) {}
#endif
