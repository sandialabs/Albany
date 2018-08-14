//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "AlbanyPeridigmOBCFunctional.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_Utils.hpp"
#include "PeridigmManager.hpp"
#include "Petra_Converters.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_TestForException.hpp"

Albany::AlbanyPeridigmOBCFunctional::AlbanyPeridigmOBCFunctional(
    const Teuchos::RCP<const Teuchos_Comm>& commT)
    : Albany::ScalarResponseFunction(commT)
{
}

Albany::AlbanyPeridigmOBCFunctional::~AlbanyPeridigmOBCFunctional() {}

unsigned int
Albany::AlbanyPeridigmOBCFunctional::numResponses() const
{
  return 1;
}

// **********************************************************************

void
Albany::AlbanyPeridigmOBCFunctional::evaluateResponse(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    Tpetra_Vector& gT)
{
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst();
  LCM::PeridigmManager& peridigmManager = *LCM::PeridigmManager::self();

  gT_nonconstView[0] = peridigmManager.obcEvaluateFunctional();

  if (commT->getRank() == 0) {
    std::cout << std::setprecision(12)
              << "\nOptimization based coupling functional value = "
              << gT_nonconstView[0] << std::endl;
  }
}

void
Albany::AlbanyPeridigmOBCFunctional::evaluateTangent(
    const double /*alpha*/,
    const double /*beta*/,
    const double /*omega*/,
    const double /*current_time*/,
    bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    ParamVec* /*deriv_p*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vx*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    Tpetra_Vector* /*gT*/,
    Tpetra_MultiVector* /*gxT*/,
    Tpetra_MultiVector* /*gpT*/)
{
  // Do Nothing
}

//! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
void
Albany::AlbanyPeridigmOBCFunctional::evaluateGradient(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*p*/,
    ParamVec* /*deriv_p*/,
    Tpetra_Vector*      gT,
    Tpetra_MultiVector* dg_dxT,
    Tpetra_MultiVector* dg_dxdotT,
    Tpetra_MultiVector* dg_dxdotdotT,
    Tpetra_MultiVector* dg_dpT)
{
  LCM::PeridigmManager& peridigmManager = *LCM::PeridigmManager::self();
  if (dg_dxT != NULL) {
    Teuchos::RCP<const Epetra_Comm> comm =
        Albany::createEpetraCommFromTeuchosComm(commT);
    Epetra_MultiVector dgdx(
        *Petra::TpetraMap_To_EpetraMap(dg_dxT->getMap(), comm),
        dg_dxT->getNumVectors(),
        false);
    Petra::TpetraMultiVector_To_EpetraMultiVector(
        Teuchos::rcp(dg_dxT, false), dgdx, comm);

    double resp = peridigmManager.obcEvaluateFunctional((dgdx)(0));
    Teuchos::RCP<Tpetra_MultiVector> dg_dxT_rcp =
        Petra::EpetraMultiVector_To_TpetraMultiVector(dgdx, commT);
    dg_dxT->assign(*dg_dxT_rcp);

    if (gT != NULL) { gT->getDataNonConst()[0] = resp; }
  } else if (gT != NULL) {
    gT->getDataNonConst()[0] = peridigmManager.obcEvaluateFunctional();
  }

  // Evaluate dg/dxdot
  if (dg_dxdotT != NULL) { dg_dxdotT->putScalar(0.0); }
  if (dg_dxdotdotT != NULL) { dg_dxdotdotT->putScalar(0.0); }

  // Evaluate dg/dp
  if (dg_dpT != NULL) { dg_dpT->putScalar(0.0); }
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::AlbanyPeridigmOBCFunctional::evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    Tpetra_MultiVector* dg_dpT)
{
  dg_dpT->putScalar(0.0);
}
