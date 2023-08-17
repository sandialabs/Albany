//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydrologyResidualCavitiesEqn.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
HydrologyResidualCavitiesEqn (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl) :
  residual (p.get<std::string> ("Cavities Eqn Residual Name"),dl->node_scalar)
{
  /*
   *  The (water) thickness equation has the following (strong) form
   *
   *     dh/dt - phi0/(rhow*g) dP/dt = m/rho_i + (h_r-h)*|u_b|/l_r - c_creep*W_c
   *
   *  where W_c = A*h*N^3 (CUBIC) or W_c = h*N/eta_i (LINEAR),
   *  h is the water thickness, P is the water pressure,
   *  phi0 is the englacial porosity, m the melting rate of the ice,
   *  h_r/l_r typical height/length of bed bumps, u_b the sliding
   *  velocity of the ice, A is the ice softness, N is the
   *  effective pressure, and c_creep is a tuning coefficient.
   *  Also, dh/dt denotes the *partial* time derivative.
   */

  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    sideSetName = p.get<std::string>("Side Set Name");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");
  }
  numNodes = dl->node_scalar->extent(1);
  numQPs   = dl->qp_scalar->extent(1);

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology Parameters");
  Teuchos::ParameterList& cav_eqn_params   = hydrology_params.sublist("Cavities Equation");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  Teuchos::ParameterList& viscosity_params = *p.get<Teuchos::ParameterList*>("LandIce Viscosity Parameters");

  unsteady = p.get<bool>("Unsteady");

  n = viscosity_params.get<double>("Glen's Law n");
  rho_i = physical_params.get<double>("Ice Density");
  h_r = cav_eqn_params.get<double>("Bed Bumps Height");
  l_r = cav_eqn_params.get<double>("Bed Bumps Length");
  c_creep = cav_eqn_params.get<double>("Creep Closure Coefficient",1.0);
  englacial_phi = cav_eqn_params.get<double>("Englacial Porosity",0.0);
  auto closure_type_N = cav_eqn_params.get("Closure Type N","Cubic");
  if (closure_type_N=="Linear") {
    closure = Linear;
    eta_i = physical_params.get<double>("Ice Viscosity");
  } else if (closure_type_N=="Cubic") {
    closure = Cubic;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (false, Teuchos::Exceptions::InvalidParameterValue,
        "Error! Unknown cavity closure type '" + closure_type_N + "'.\n"
        "       Valid options are: Linear, Cubic.\n");
  }
  if (englacial_phi>0 && unsteady) {
    has_p_dot = true;
    rho_w = physical_params.get<double>("Water Density");
    g = physical_params.get<double>("Gravity Acceleration");
  } else {
    has_p_dot = false;
  }
  use_eff_cavity = cav_eqn_params.get<bool>("Use Effective Cavity",true);

  use_melting = cav_eqn_params.get<bool>("Use Melting", false);

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The residual has 5 terms (here in strong form, without signs), with the following units:
   *
   *  1) h_t              [m s^-1 ]
   *  2) phi0/(rhow*g)P_t [km s^-1]
   *  3) m/rho_i          [m yr^-1]
   *  4) c_creep*W_c      [m k^3 s^-1 ]
   *  5) (h_r-h)*|u|/l_r  [m yr^-1]
   *
   * We decide to uniform all terms to have units [m yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) scaling_h_t*h_t                scaling_h_t = yr_to_s
   *  2) scaling_P_t*phi0/(rhow*g)P_t   scaling_P_t = 10^3
   *  3) m/rho_i                        (no scaling)
   *  4) scalinc_c*c_creep*W_c          scaling_c = yr_to_s * k^n
   *  5) (h_r-h)*|u|/l_r                (no scaling)
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   * Note: the k^n (n=3 for Cubic, n=1 for Linear) comes from N being in kPa and
   *         - Cubic: A in Pa^-3 s^-1.
   *         - Cubic: eta in Pa^- s^-1.
   */

  // We can solve this equation as a nodal equation
  nodal_equation = cav_eqn_params.isParameter("Nodal") ? cav_eqn_params.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout;
  if (nodal_equation) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;

    BF        = PHX::MDField<const RealType>(p.get<std::string> ("BF Name"), dl->node_qp_scalar);
    w_measure = PHX::MDField<const MeshScalarT>(p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar);

    this->addDependentField(BF);
    this->addDependentField(w_measure);
  }

  h            = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Thickness Variable Name"),     layout);
  N            = PHX::MDField<const ScalarT>(p.get<std::string> ("Effective Pressure Variable Name"),  layout);
  u_b          = PHX::MDField<const IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
  ice_softness = PHX::MDField<const TempScalarT>(p.get<std::string>("Ice Softness Variable Name"), dl->cell_scalar2);
  this->addDependentField(h);
  this->addDependentField(N);
  if (use_melting) {
    m  = PHX::MDField<const ScalarT>(p.get<std::string> ("Melting Rate Variable Name"), layout);
    this->addDependentField(m);
  }
  this->addDependentField(u_b);
  this->addDependentField(ice_softness);

  if (unsteady) {
    h_dot = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Thickness Dot Variable Name"), layout);
    this->addDependentField(h_dot);
    if (has_p_dot) {
      P_dot = PHX::MDField<const ScalarT>(p.get<std::string>("Water Pressure Dot Variable Name"), layout);
      this->addDependentField(P_dot);
    }
  }

  this->setName("HydrologyResidualCavitiesEqn"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (closure==Linear, std::runtime_error,
      "Error! I haven't implemented Ian Hewitt's 2011 linear creep closure for the Stokes coupled case.\n");
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  // h' = W_O - W_C = (m/rho_i + u_b*(h_b-h)/l_b) - c_creep*AhN^n
  ScalarT res_node, res_qp, zero(0.0);

  // Zero out, to avoid leaving stuff from previous workset!
  residual.deep_copy(ScalarT(0.0));

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) return;

  // Note: the '1000^n' is to convert the ice softness in kPa^-n s^-1, so that the kPa
  //       cancel out with N, and the residual is in m/yr
  double yr_to_s = 365.25*24*3600;
  double scaling_h_t = yr_to_s;
  double C = c_creep * yr_to_s * pow(1000,n);
  double phi0 = has_p_dot ? englacial_phi / (1000*rho_w*g) : 0.0;

  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the local data of cell
    const int cell = sideSet.ws_elem_idx.h_view(sideSet_idx);

    for (unsigned int node=0; node < numNodes; ++node) {
      res_node = 0;
      if (nodal_equation) {
        res_node = (use_melting ? m(sideSet_idx,node)/rho_i : zero)
                 + (use_eff_cavity ? (h_r - h(sideSet_idx,node))*u_b(sideSet_idx,node)/l_r
                                   : ScalarT(h_r*u_b(sideSet_idx,node)))
                 - C*h(sideSet_idx,node)*ice_softness(cell)*std::pow(N(sideSet_idx,node),n)
                 - (unsteady ? scaling_h_t*h_dot(sideSet_idx,node) : zero)
                 + (has_p_dot ? -phi0*P_dot(sideSet_idx,node) : zero);
      } else {
        for (unsigned int qp=0; qp < numQPs; ++qp) {
          res_qp = (use_melting ? m(sideSet_idx,qp)/rho_i : zero)
                 + (use_eff_cavity ? (h_r - h(sideSet_idx,qp))*u_b(sideSet_idx,qp)/l_r
                                   : ScalarT(h_r*u_b(sideSet_idx,qp)))
                 - C*h(sideSet_idx,qp)*ice_softness(sideSet_idx)*std::pow(N(sideSet_idx,qp),n)
                 - (unsteady ? scaling_h_t*h_dot(sideSet_idx,qp) : zero)
                 + (has_p_dot ? -phi0*P_dot(sideSet_idx,qp) : zero);

          res_node += res_qp * BF(sideSet_idx,node,qp) * w_measure(sideSet_idx,qp);
        }
      }

      residual (sideSet_idx,node) = res_node;
    }
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  // h' = W_O - W_C = (m/rho_i + u_b*(h_b-h)/l_b) - AhN^n
  ScalarT res_node, res_qp, zero(0.0);

  // Note: the '1000^n' is to convert the ice softness in kPa^-n s^-1, so that the kPa
  //       cancel out with N, and the residual is in m/yr
  double yr_to_s = 365.25*24*3600;
  double scaling_h_t = yr_to_s;
  double C = c_creep * yr_to_s * (closure==Cubic ? pow(1000,n) : 1);
  double phi0 = has_p_dot ? englacial_phi / (1000*rho_w*g) : 0.0;
  double etai = eta_i/1000; // Convert to kPa s

  for (unsigned int cell=0; cell < workset.numCells; ++cell) {
    for (unsigned int node=0; node < numNodes; ++node) {
      res_node = 0;
      if (nodal_equation) {
        res_node = (use_melting ? m(cell,node)/rho_i : zero)
                 + (use_eff_cavity ? (h_r - h(cell,node))*u_b(cell,node)/l_r
                                   : ScalarT(h_r*u_b(cell,node)))
                 - (unsteady ? scaling_h_t*h_dot(cell,node) : zero)
                 + (has_p_dot ? -phi0*P_dot(cell,node) : zero);
        switch (closure) {
          case Cubic:
            res_node -= C*h(cell,node)*ice_softness(cell)*std::pow(N(cell,node),n);
            break;
          case Linear:
            res_node -= C*h(cell,node)*N(cell,node)/etai;
            break;
        }
      } else {
        for (unsigned int qp=0; qp < numQPs; ++qp) {
          res_qp = (use_melting ? m(cell,qp)/rho_i : zero)
                 + (use_eff_cavity ? (h_r - h(cell,qp))*u_b(cell,qp)/l_r
                                   : ScalarT(h_r*u_b(cell,qp)))
                 - (unsteady ? scaling_h_t*h_dot(cell,qp) : zero)
                 + (has_p_dot ? -phi0*P_dot(cell,qp) : zero);
          switch (closure) {
            case Cubic:
              res_qp -= C*h(cell,qp)*ice_softness(cell)*std::pow(N(cell,qp),n);
              break;
            case Linear:
              res_qp -= C*h(cell,qp)*N(cell,qp)/eta_i;
              break;
          }

          res_node += res_qp * BF(cell,node,qp) * w_measure(cell,qp);
        }
      }
      residual (cell,node) = res_node;
    }
  }
}

} // Namespace LandIce
