//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydrologyResidualMassEqn.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyResidualMassEqn<EvalT, Traits>::
HydrologyResidualMassEqn (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
  BF        (p.get<std::string> ("BF Name"), dl->node_qp_scalar),
  GradBF    (p.get<std::string> ("Gradient BF Name"), dl->node_qp_gradient),
  w_measure (p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar),
  q         (p.get<std::string> ("Water Discharge Variable Name"), dl->qp_gradient),
  omega     (p.get<std::string> ("Surface Water Input Variable Name"), dl->qp_scalar),
  residual  (p.get<std::string> ("Mass Eqn Residual Name"),dl->node_scalar)
{
  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  numQPs   = eval_on_side ? dl->qp_scalar->dimension(2) : dl->qp_scalar->dimension(1);
  numNodes = eval_on_side ? dl->node_scalar->dimension(2) : dl->node_scalar->dimension(1);
  numDims  = eval_on_side ? dl->qp_gradient->dimension(3) : dl->qp_gradient->dimension(2);

  if (eval_on_side) {
    metric = PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>(p.get<std::string>("Metric Name"),dl->qp_tensor);
    this->addDependentField(metric);
  }

  this->addEvaluatedField(residual);

  // Setting parameters
  auto& hydro_params    = *p.get<Teuchos::ParameterList*>("LandIce Hydrology Parameters");
  auto& mass_eqn_params = hydro_params.sublist("Mass Equation");
  auto& physical_params = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_w       = physical_params.get<double>("Water Density", 1028.0);
  use_melting = mass_eqn_params.get<bool>("Use Melting", false);

  unsteady = p.get<bool>("Unsteady");
  has_h_till = p.get<bool>("Has Till Storage");

  if (unsteady) {
    h_dot = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Thickness Dot Variable Name"), dl->qp_scalar);
    this->addDependentField(h_dot);

    if (has_h_till) {
      h_till_dot = PHX::MDField<const ScalarT>(p.get<std::string> ("Till Water Storage Dot Variable Name"), dl->qp_scalar);
      this->addDependentField(h_till_dot);
    }
  }

  if (use_melting) {
    mass_lumping = mass_eqn_params.isParameter("Lump Mass") ? mass_eqn_params.get<bool>("Lump Mass") : false;
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
  this->addDependentField(GradBF);
  this->addDependentField(w_measure);
  this->addDependentField(q);
  this->addDependentField(omega);

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology space/time unit are m/s.
   *
   * The residual has 4 terms (here in strong form, without signs), with the following units:
   *
   *  1) dh/dt            [m  s^-1  ]
   *  2) div(q)           [mm s^-1  ]     ([q] = m^2/s, but mesh is in km, so [div] = 1/km)
   *  3) m/rho_w          [m  yr^-1 ]
   *  4) omega            [mm day^-1]
   *
   * where q=k*h^alpha*|gradPhi|^(beta-2)*gradPhi.
   * Notice that [q] = m^2/s, but the mesh is in km, hence [div(q)] = 1e-3 m/s
   * We decide to uniform all terms to have units [m yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) scaling_h_dot*dh/dt    scaling_h_dot = yr_to_s
   *  2) scaling_q*div(q)       scaling_q     = 1e-3*yr_to_s
   *  3) m/rho_w                (no scaling)
   *  4) scaling_omega*omega    scaling_omega = 1e-3*yr_to_d
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   * and   yr_to_d=365.25 (the number of days in a year)
   */
  double yr_to_d  = 365.25;
  double d_to_s   = 24*3600;
  double yr_to_s  = yr_to_d*d_to_s;
  scaling_omega   = 1e-3*yr_to_d;
  scaling_h_dot   = yr_to_s;
  scaling_q       = 1e-3*yr_to_s;

  this->setName("HydrologyResidualMassEqn"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidualMassEqn<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void HydrologyResidualMassEqn<EvalT, Traits>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  // Zero out, to avoid leaving stuff from previous workset!
  residual.deep_copy(ScalarT(0.));

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  ScalarT res_qp, res_node;
  const auto& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (unsigned int node=0; node < numNodes; ++node) {
      res_node = 0;
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        res_qp = scaling_omega*omega(cell,side,qp);
        if (unsteady) {
          res_qp -= scaling_h_dot*h_dot(cell,side,qp);
          if (has_h_till) {
            res_qp -= scaling_h_dot*h_till_dot(cell,side,qp);
          }
        }

        if (use_melting && !mass_lumping) {
          res_qp += m(cell,side,qp)/rho_w;
        }

        res_qp *= BF(cell,side,node,qp);

        for (unsigned int idim=0; idim<numDims; ++idim) {
          for (unsigned int jdim=0; jdim<numDims; ++jdim) {
            res_qp += scaling_q*q(cell,side,qp,idim) * metric(cell,side,qp,idim,jdim) * GradBF(cell,side,node,qp,jdim);
          }
        }

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
void HydrologyResidualMassEqn<EvalT, Traits>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT res_qp, res_node;
  for (unsigned int cell=0; cell < workset.numCells; ++cell) {
    for (unsigned int node=0; node < numNodes; ++node) {
      res_node = 0;
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        res_qp = scaling_omega*omega(cell,qp);
        if (unsteady) {
          res_qp -= scaling_h_dot*h_dot(cell,qp);
          if (has_h_till) {
            res_qp -= scaling_h_dot*h_till_dot(cell,qp);
          }
        }

        if (use_melting && !mass_lumping) {
          res_qp += m(cell,qp)/rho_w;
        }

        res_qp *= BF(cell,node,qp);

        for (unsigned int dim=0; dim<numDims; ++dim) {
          res_qp += scaling_q*q(cell,qp,dim) * GradBF(cell,node,qp,dim);
        }

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
