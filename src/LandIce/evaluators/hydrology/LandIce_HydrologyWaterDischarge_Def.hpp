//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "LandIce_HydrologyWaterDischarge.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits>
HydrologyWaterDischarge<EvalT, Traits>::
HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  gradPhi (p.get<std::string> ("Hydraulic Potential Gradient Variable Name"), dl->qp_gradient),
  h       (p.get<std::string> ("Water Thickness Variable Name"), dl->qp_scalar),
  q       (p.get<std::string> ("Water Discharge Variable Name"), dl->qp_gradient)
{
  /*
   *  The water discharge follows the following Darcy-like form
   *
   *     q = - k * h^alpha * |grad(Phi)|^(beta-2) * grad(Phi)
   *
   *  where q is the water discharge, h the water thickness, k a transmissivity constant,
   *  phi is the hydraulic poential, and alpha/beta are two constants, with requirements
   *  alpha>1, beta>1. The units of q follow from those of the mesh, h, k and Phi.
   *  We assume h is in [m], Phi in [kPa], the mesh is in [km], and k has units
   *     [k] =  m^(2*beta-alpha) s^(2*beta-3) kg^(1-beta).
   *  In the common case of beta=2, alpha=1, we have [k] = m^3 s kg^-1
   *  Putting everything togeter, we get
   *     [q] = m^2/s
   */

  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  numQPs  = eval_on_side ? dl->qp_gradient->extent(2) : dl->qp_gradient->extent(1);
  numDim  = eval_on_side ? dl->qp_gradient->extent(3) : dl->qp_gradient->extent(2);

  this->addDependentField(gradPhi);
  this->addDependentField(h);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology");
  Teuchos::ParameterList& darcy_law_params = hydrology_params.sublist("Darcy Law");

  k_0   = darcy_law_params.get<double>("Transmissivity");
  alpha = darcy_law_params.get<double>("Water Thickness Exponent");
  beta  = darcy_law_params.get<double>("Potential Gradient Norm Exponent");

  TEUCHOS_TEST_FOR_EXCEPTION (
      beta<=1, Teuchos::Exceptions::InvalidParameter,
      "Error! 'Darcy Law: Potential Gradient Norm Exponent' must be larger than 1.0.\n");

  if (beta==2.0) {
    needsGradPhiNorm = false;
  } else {
    needsGradPhiNorm = true;
  }

  if (needsGradPhiNorm) {
    gradPhiNorm = decltype(gradPhiNorm)(p.get<std::string>("Hydraulic Potential Gradient Norm Variable Name"), dl->qp_scalar);
    this->addDependentField(gradPhiNorm);
  }

  regularize = darcy_law_params.get<bool>("Regularize With Continuation", false);
  if (regularize)
  {
    regularizationParam = PHX::MDField<ScalarT,Dim>(p.get<std::string>("Regularization Parameter Name"),dl->shared_param);
    this->addDependentField(regularizationParam);
  }

  this->setName("HydrologyWaterDischarge"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT regularization(0.0);
  if (regularize) {
    regularization = regularizationParam(0);
  }
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);

  static ScalarT printedReg = -1;
  if (printedReg!=regularization) {
    *output << "[HydrologyWaterDischarge" << PHX::print<EvalT>() << "] reg = " << regularization << "\n";
    printedReg = regularization;
  }

  if (needsGradPhiNorm) {
    double grad_norm_exponent = beta - 2.0;
    ScalarT hpow(0.0);
    for (unsigned int cell=0; cell < workset.numCells; ++cell) {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        hpow = h(cell,qp)*std::pow(std::abs(h(cell,qp)),alpha-1);
        for (unsigned int dim(0); dim<numDim; ++dim) {
          q(cell,qp,dim) = -k_0 * (hpow+regularization)
                                * std::pow(gradPhiNorm(cell,qp),grad_norm_exponent)
                                * gradPhi(cell,qp,dim);
        }
      }
    }
  } else {
    ScalarT hpow(0.0);
    for (unsigned int cell=0; cell < workset.numCells; ++cell) {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        hpow = h(cell,qp)*std::pow(std::abs(h(cell,qp)),alpha-1);
        for (unsigned int dim(0); dim<numDim; ++dim) {
          q(cell,qp,dim) = - k_0 * (hpow+regularization) * gradPhi(cell,qp,dim);
        }
      }
    }
  }
}

template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  ScalarT regularization(0.0);
  if (regularize) {
    regularization = regularizationParam(0);
  }

  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
  static ScalarT printedReg = -1;
  if (printedReg!=regularization) {
    *output << "[HydrologyWaterDischarge<" << PHX::print<EvalT>() << ">] reg = " << regularization << "\n";
    printedReg = regularization;
  }

  const auto& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    if (needsGradPhiNorm) {
      double grad_norm_exponent = beta - 2.0;
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        for (unsigned int dim(0); dim<numDim; ++dim) {
          q(cell,side,qp,dim) = -k_0 * (std::pow(h(cell,side,qp),alpha)+regularization)
                                     * std::pow(gradPhiNorm(cell,side,qp),grad_norm_exponent)
                                     * gradPhi(cell,side,qp,dim);
        }
      }
    } else {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        for (unsigned int dim(0); dim<numDim; ++dim) {
          q(cell,side,qp,dim) = -k_0 * (std::pow(h(cell,side,qp),alpha)+regularization) * gradPhi(cell,side,qp,dim);
        }
      }
    }
  }
}

} // Namespace LandIce
