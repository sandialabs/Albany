//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_HydrologyResidualTillStorageEqn.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyResidualTillStorageEqn<EvalT, Traits>::
HydrologyResidualTillStorageEqn (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
  BF         (p.get<std::string> ("BF Name"), dl->node_qp_scalar),
  w_measure  (p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar),
  omega      (p.get<std::string> ("Surface Water Input Variable Name"), dl->qp_scalar),
  h_till_dot (p.get<std::string> ("Till Water Storage Dot Variable Name"), dl->qp_scalar),
  residual   (p.get<std::string> ("Till Water Storage Eqn Residual Name"),dl->node_scalar)
{
  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  numNodes = eval_on_side ? dl->node_scalar->extent(2) : dl->node_scalar->extent(1);
  numQPs   = eval_on_side ? dl->qp_scalar->extent(2)   : dl->qp_scalar->extent(1);

  this->addEvaluatedField(residual);

  // Setting parameters
  auto& hydro_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology Parameters");
  auto& phys_params  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_w       = phys_params.get<double>("Water Density", 1028.0);
  use_melting = hydro_params.get<bool>("Use Melting In Conservation Of Mass", false);
  C_drain     = hydro_params.get<double>("Till Water Storage Drain Rate", -1.0);
  TEUCHOS_TEST_FOR_EXCEPTION (C_drain<=0, Teuchos::Exceptions::InvalidParameter,
                              "Error! The till water storage drain rate must be positive.\n");

  if (use_melting) {
    mass_lumping = hydro_params.isParameter("Lump Mass In Mass Equation") ? hydro_params.get<bool>("Lump Mass In Mass Equation") : false;
  } else {
    mass_lumping = false;
  }

  Teuchos::RCP<PHX::DataLayout> layout;
  if (mass_lumping) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  if (use_melting) {
    m = PHX::MDField<const ScalarT>(p.get<std::string> ("Melting Rate Variable Name"), layout);
    this->addDependentField(m);
  }

  this->addDependentField(BF);
  this->addDependentField(w_measure);
  this->addDependentField(h_till_dot);
  this->addDependentField(omega);

  /*
   * Scalings, needed to account for different units
   *
   * The residual has 4 terms (here in strong form, without signs), with the following units:
   *
   *  1) dh_till/dt       [m  s^-1  ]
   *  2) m/rho_w          [m  yr^-1 ]
   *  3) omega            [mm day^-1]
   *  4) C_drain          [mm yr^-1 ]
   *
   * We decide to uniform all terms to have units [m yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) scaling_h_dot*dh_till/dt   scaling_h_dot = yr_to_s
   *  2) m/rho_w                    (no scaling)
   *  3) scaling_omega*omega        scaling_omega = 1e-3*yr_to_d
   *  4) scaling_C*C_drain          scaling_C = 1e-3
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   * and   yr_to_d=365.25 (the number of days in a year)
   */
  double yr_to_d  = 365.25;
  double yr_to_s  = yr_to_d*24*3600;
  scaling_omega   = 1e-3*yr_to_d;
  scaling_h_dot   = yr_to_s;
  C_drain *= 1e-3;

  this->setName("HydrologyResidualTillStorageEqn"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidualTillStorageEqn<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void HydrologyResidualTillStorageEqn<EvalT, Traits>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  // Zero out, to avoid leaving stuff from previous workset!
  residual.deep_copy(ScalarT(0.));

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  ScalarT res_qp, res_node;
  const auto& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (unsigned int node=0; node < numNodes; ++node) {
      res_node = 0;
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        res_qp = scaling_omega*omega(cell,side,qp) - C_drain - scaling_h_dot*h_till_dot(cell,side,qp);

        if (use_melting && !mass_lumping) {
          res_qp += m(cell,side,qp)/rho_w;
        }

        res_qp *= BF(cell,side,node,qp);

        res_node += res_qp * w_measure(cell,side,qp);
      }

      if (use_melting && mass_lumping) {
        res_node += m(cell,side,node)/rho_w;
      }

      residual (cell,side,node) = res_node;
    }
  }
}

template<typename EvalT, typename Traits>
void HydrologyResidualTillStorageEqn<EvalT, Traits>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT res_qp, res_node;
  for (unsigned int cell=0; cell < workset.numCells; ++cell) {
    for (unsigned int node=0; node < numNodes; ++node) {
      res_node = 0;
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        res_qp = scaling_omega*omega(cell,qp) - C_drain - scaling_h_dot*h_till_dot(cell,qp);

        if (use_melting && !mass_lumping) {
          res_qp += m(cell,qp)/rho_w;
        }

        res_qp *= BF(cell,node,qp);

        res_node += res_qp * w_measure(cell,qp);
      }

      if (use_melting && mass_lumping) {
        res_node += m(cell,node)/rho_w;
      }

      residual (cell,node) = res_node;
    }
  }
}

} // Namespace LandIce
