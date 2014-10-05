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
evaluateTangentT (const double alpha, 
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
evaluateGradientT (const double current_time,
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
