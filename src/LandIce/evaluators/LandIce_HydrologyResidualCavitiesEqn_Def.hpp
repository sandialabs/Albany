//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

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
   *     dh/dt - phi0/(rhow*g) dP/dt = m/rho_i + (h_r-h)*|u_b|/l_r - c_creep*A*h*N^3
   *
   *  where h is the water thickness, P is the water pressure,
   *  phi0 is the englacial porositi, m the melting rate of the ice,
   *  h_r/l_r typical height/length of bed bumps, u_b the sliding
   *  velocity of the ice, A is the ice softness, N is the
   *  effective pressure, and c_creep is a tuning coefficient.
   *  Also, dh/dt denotes the *partial* time derivative.
   */

  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(2);
    numQPs   = dl->qp_scalar->dimension(2);

    sideSetName = p.get<std::string>("Side Set Name");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
  }

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  unsteady = p.get<bool>("Unsteady");

  rho_i = physical_params.get<double>("Ice Density");
  h_r = hydrology_params.get<double>("Bed Bumps Height");
  l_r = hydrology_params.get<double>("Bed Bumps Length");
  c_creep = hydrology_params.get<double>("Creep Closure Coefficient",1.0);
  phi0 = hydrology_params.get<double>("Englacial Porosity",0.0);
  if (phi0>0 && unsteady) {
    has_p_dot = true;
    double rho_w = physical_params.get<double>("Water Density");
    double g = physical_params.get<double>("Gravity Acceleration");
    phi0 /= (rho_w*g);
  } else {
    has_p_dot = false;
  }

  use_melting = hydrology_params.get<bool>("Use Melting In Cavities Equation", false);

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The residual has 5 terms (here in strong form, without signs), with the following units:
   *
   *  1) h_t              [m s^-1 ]
   *  2) phi0/(rhow*g)P_t [km s^-1]
   *  3) m/rho_i          [m yr^-1]
   *  4) c_creep*A*h*N^3  [m s^-1 ]
   *  5) (h_r-h)*|u|/l_r  [m yr^-1]
   *
   * We decide to uniform all terms to have units [m yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) scaling_h_t*h_t                scaling_h_t = yr_to_s
   *  2) scaling_P_t*phi0/(rhow*g)P_t   scaling_P_t = 10^3
   *  3) m/rho_i                        (no scaling)
   *  4) c_creep*scaling_A*A*h*N^3      scaling_A = yr_to_s
   *  5) (h_r-h)*|u|/l_r                (no scaling)
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   */

  double yr_to_s = 365.25*24*3600;

  scaling_h_t = yr_to_s;
  c_creep *= yr_to_s;
  phi0 *= 1e3;

  // We can solve this equation as a nodal equation
  nodal_equation = hydrology_params.isParameter("Cavities Equation Nodal") ? hydrology_params.get<bool>("Cavities Equation Nodal") : false;
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

  this->setName("HydrologyResidualCavitiesEqn"+PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(N,fm);
  if (use_melting) {
    this->utils.setFieldData(m,fm);
  }
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(ice_softness,fm);
  if (unsteady) {
    this->utils.setFieldData(h_dot,fm);
    if (has_p_dot) {
      this->utils.setFieldData(P_dot,fm);
    }
  }
  if (!nodal_equation) {
    this->utils.setFieldData(BF,fm);
    this->utils.setFieldData(w_measure,fm);
  }
  this->utils.setFieldData(residual,fm);
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  // h' = W_O - W_C = (m/rho_i + u_b*(h_b-h)/l_b) - AhN^n
  ScalarT res_node, res_qp, zero(0.0);

  // Zero out, to avoid leaving stuff from previous workset!
  residual.deep_copy(ScalarT(0.0));

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int node=0; node < numNodes; ++node) {
      res_node = 0;
      if (nodal_equation) {
        res_node = (use_melting ? m(cell,side,node)/rho_i : zero)
                 + (h_r - h(cell,side,node))*u_b(cell,side,node)/l_r
                 - c_creep*h(cell,side,node)*ice_softness(cell)*std::pow(N(cell,side,node),3)
                 - (unsteady ? scaling_h_t*h_dot(cell,side,node) : zero)
                 + (has_p_dot ? -phi0*P_dot(cell,side,node) : zero);
      } else {
        for (int qp=0; qp < numQPs; ++qp) {
          res_qp = (use_melting ? m(cell,side,qp)/rho_i : zero)
                 + (h_r - h(cell,side,qp))*u_b(cell,side,qp)/l_r
                 - c_creep*h(cell,side,qp)*ice_softness(cell,side)*std::pow(N(cell,side,qp),3)
                 - (unsteady ? scaling_h_t*h_dot(cell,side,qp) : zero)
                 + (has_p_dot ? -phi0*P_dot(cell,side,qp) : zero);

          res_node += res_qp * BF(cell,side,node,qp) * w_measure(cell,side,qp);
        }
      }

      residual (cell,side,node) = res_node;
    }
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyResidualCavitiesEqn<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  // h' = W_O - W_C = (m/rho_i + u_b*(h_b-h)/l_b) - AhN^n
  ScalarT res_node, res_qp, zero(0.0);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      res_node = 0;
      if (nodal_equation) {
        res_node = (use_melting ? m(cell,node)/rho_i : zero)
                 + (h_r - h(cell,node))*u_b(cell,node)/l_r
                 - c_creep*h(cell,node)*ice_softness(cell)*std::pow(N(cell,node),3)
                 - (unsteady ? scaling_h_t*h_dot(cell,node) : zero)
                 + (has_p_dot ? -phi0*P_dot(cell,node) : zero);
      } else {
        for (int qp=0; qp < numQPs; ++qp) {
          res_qp = (use_melting ? m(cell,qp)/rho_i : zero)
                 + (h_r - h(cell,qp))*u_b(cell,qp)/l_r
                 - c_creep*h(cell,qp)*ice_softness(cell)*std::pow(N(cell,qp),3)
                 - (unsteady ? scaling_h_t*h_dot(cell,qp) : zero)
                 + (has_p_dot ? -phi0*P_dot(cell,qp) : zero);

          res_node += res_qp * BF(cell,node,qp) * w_measure(cell,qp);
        }
      }
      residual (cell,node) = res_node;
    }
  }
}

} // Namespace LandIce
