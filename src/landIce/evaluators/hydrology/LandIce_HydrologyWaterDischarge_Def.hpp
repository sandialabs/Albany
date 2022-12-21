//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "LandIce_HydrologyWaterDischarge.hpp"
#include <stdexcept>

//uncomment the following line if you want debug output to be printed to screen
#define DEBUG_OUTPUT

namespace LandIce
{

template<typename EvalT, typename Traits>
HydrologyWaterDischarge<EvalT, Traits>::
HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  gradPhi (p.get<std::string> ("Hydraulic Potential Gradient Variable Name"), dl->qp_gradient),
  h       (p.get<std::string> ("Water Thickness Variable Name"), dl->qp_scalar),
  k_param (p.get<std::string> ("Transmissivity Parameter Name"), dl->shared_param),
  q       (p.get<std::string> ("Water Discharge Variable Name"), dl->qp_gradient)
{
  /*
   *  The water discharge follows the following Darcy-like form
   *
   *     q = - k * h^alpha * |grad(Phi)|^(beta-2) * grad(Phi)
   *
   *  where q is the water discharge, h the water thickness, k a transmissivity constant,
   *  phi is the hydraulic potential, and alpha/beta are two constants, with requirements
   *  alpha>1, beta>1. The units of q follow from those of the mesh, h, k and Phi.
   *  We assume h is in [m], Phi in [kPa], the mesh is in [km], and k has units
   *     [k] =  m^(2*beta-alpha) s^(2*beta-3) kg^(1-beta).
   *  In the common case of beta=2, alpha=1, we have [k] = m^3 s kg^-1
   *  Putting everything together, we get
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

  numQPs  = dl->qp_gradient->extent(1);
  numDim  = dl->qp_gradient->extent(2);

  this->addDependentField(gradPhi);
  this->addDependentField(h);
  this->addDependentField(k_param);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology");
  Teuchos::ParameterList& darcy_law_params = hydrology_params.sublist("Darcy Law");

  // k_0   = darcy_law_params.get<double>("Transmissivity");
  alpha = darcy_law_params.get<double>("Water Thickness Exponent");
  beta  = darcy_law_params.get<double>("Potential Gradient Norm Exponent");

  TEUCHOS_TEST_FOR_EXCEPTION (
      beta<=1, Teuchos::Exceptions::InvalidParameter,
      "Error! 'Darcy Law: Potential Gradient Norm Exponent' must be larger than 1.0.\n"
      "   Input value: " + std::to_string(beta) + "\n");

  if (beta==2.0) {
    needsGradPhiNorm = false;
  } else {
    needsGradPhiNorm = true;
  }

  if (needsGradPhiNorm) {
    gradPhiNorm = decltype(gradPhiNorm)(p.get<std::string>("Hydraulic Potential Gradient Norm Variable Name"), dl->qp_scalar);
    this->addDependentField(gradPhiNorm);
  }

  auto& reg_pl = darcy_law_params.sublist("Regularization");
  auto type = reg_pl.get<std::string>("Regularization Type","None");
  if (type=="None") {
    reg_type = NONE;
    regularization = 0.0;
  } else if (type=="Given Value") {
    reg_type = GIVEN_VALUE;
    regularization = reg_pl.get<double>("Regularization Value");
  } else if (type=="Given Parameter") {
    reg_type = GIVEN_PARAMETER;
    auto pname = reg_pl.get<std::string>("Regularization Parameter Name");
    regularizationParam = PHX::MDField<ScalarT,Dim>(pname, dl->shared_param);
    this->addDependentField(regularizationParam);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Error! Invalid choice for 'Regularization Type'. Valid options: 'Given Parameter', 'Given Value', 'None'.\n");
  }

  // Force them to be printed the 1st time the evaluator is called
  printedReg = -1.0;
  printedKappa = -1.0;

  this->setName("HydrologyWaterDischarge"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  if (reg_type==GIVEN_PARAMETER) {
    regularization = regularizationParam(0);
  }

  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);

  auto k_0 = k_param(0);
#ifdef DEBUG_OUTPUT
  // if (printedReg!=regularization) {
  //   *output << "[HydrologyWaterDischarge" << PHX::print<EvalT>() << "] reg = " << regularization << "\n";
  //   printedReg = regularization;
  // }
  if (printedKappa!=k_0) {
    *output << "[HydrologyWaterDischarge" << PHX::print<EvalT>() << "] kappa = " << k_0 << "\n";
    printedKappa = k_0;
  }
#endif

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
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) return;

  if (reg_type==GIVEN_PARAMETER) {
    regularization = regularizationParam(0);
  }

  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);

  auto k_0 = k_param(0);
  TEUCHOS_TEST_FOR_EXCEPTION (
      k_0<=0, Teuchos::Exceptions::InvalidParameter,
      "Error in LandIce::HydrologyWaterDischarge: 'Transmissivity' must be > 0.\n"
      "   Input value: " + std::to_string(Albany::ADValue(k_0)) + "\n");
#ifdef DEBUG_OUTPUT
  // if (printedReg!=regularization) {
  //   *output << "[HydrologyWaterDischarge<" << PHX::print<EvalT>() << ">] reg = " << regularization << "\n";
  //   printedReg = regularization;
  // }
  if (printedKappa!=k_0) {
    *output << "[HydrologyWaterDischarge" << PHX::print<EvalT>() << "] kappa = " << k_0 << "\n";
    printedKappa = k_0;
  }
#endif


  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    if (needsGradPhiNorm) {
      double grad_norm_exponent = beta - 2.0;
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        for (unsigned int dim(0); dim<numDim; ++dim) {
          q(sideSet_idx,qp,dim) = -k_0 * (std::pow(h(sideSet_idx,qp),alpha)+regularization)
                                     * std::pow(gradPhiNorm(sideSet_idx,qp),grad_norm_exponent)
                                     * gradPhi(sideSet_idx,qp,dim);
        }
      }
    } else {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        for (unsigned int dim(0); dim<numDim; ++dim) {
          q(sideSet_idx,qp,dim) = -k_0 * (std::pow(h(sideSet_idx,qp),alpha)+regularization) * gradPhi(sideSet_idx,qp,dim);
        }
      }
    }
  }
}

} // Namespace LandIce
