//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

inline std::shared_ptr<CP::FlowParameterBase>
CP::flowParameterFactory(CP::FlowRuleType type_flow_rule)
{
  using FPUP = std::shared_ptr<CP::FlowParameterBase>;

  switch (type_flow_rule) {

  default:
    std::cerr << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "ERROR: Unknown flow rule\n";
    exit(1);
    break;

  case FlowRuleType::POWER_LAW:
    return FPUP(new CP::PowerLawFlowParameters());
    break;

  case FlowRuleType::POWER_LAW_DRAG:
    return FPUP(new CP::PowerLawDragFlowParameters());
    break;

  case FlowRuleType::THERMAL_ACTIVATION:
    return FPUP(new CP::ThermalActivationFlowParameters());
    break;

  case FlowRuleType::UNDEFINED:
    return FPUP(new CP::NoFlowParameters());
    break;

  }

  return FPUP(nullptr);
}


//
// Power law flow rule
//
template<typename ScalarT>
ScalarT
CP::PowerLawFlowRule<ScalarT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ScalarT const & shear,
  ScalarT const & slip_resistance)
{
  ScalarT
  rate_slip{0.};

  // Material properties
  RealType const
  m = pflow_parameters->flow_params_[pflow_parameters->param_map_["Rate Exponent"]];

  RealType const
  g0 = pflow_parameters->flow_params_[pflow_parameters->param_map_["Reference Slip Rate"]];

  ScalarT const
  ratio_stress = shear / slip_resistance;

  // Compute slip increment
  rate_slip = g0 * std::pow(std::fabs(ratio_stress), m-1) * ratio_stress;

  return rate_slip;
}

//
// Thermally-activated flow rule
//
template<typename ScalarT>
ScalarT
CP::ThermalActivationFlowRule<ScalarT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ScalarT const & shear,
  ScalarT const & slip_resistance)
{
  ScalarT
  rate_slip{0.};

  // Material properties
  
  // Compute slip increment

  return rate_slip;
}

//
// Power law with Drag flow rule
//
template<typename ScalarT>
ScalarT
CP::PowerLawDragFlowRule<ScalarT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ScalarT const & shear,
  ScalarT const & slip_resistance)
{     

  // Material properties
  RealType const
  m = pflow_parameters->flow_params_[pflow_parameters->param_map_["Rate Exponent"]];

  RealType const
  g0 = pflow_parameters->flow_params_[pflow_parameters->param_map_["Reference Slip Rate"]];

  RealType const
  drag_term = pflow_parameters->flow_params_[pflow_parameters->param_map_["Drag Coefficient"]];

  ScalarT const
  ratio_stress = shear / slip_resistance;

  // Compute drag term
  ScalarT const
  viscous_drag = ratio_stress / drag_term;

  RealType const
  min_tol = std::pow(2.0 * std::numeric_limits<RealType>::min(), 0.5 / m);

  RealType const
  machine_eps = std::numeric_limits<RealType>::epsilon();

  bool const
  finite_power_law = std::fabs(ratio_stress) > min_tol;

  // carry derivative info from ratio_stress
  ScalarT
  power_law{0.0 * ratio_stress};

  if (finite_power_law == true) {
    power_law = std::pow(std::fabs(ratio_stress), m - 1) * ratio_stress;
  }

  ScalarT
  pl_vd_ratio = drag_term * std::pow(ratio_stress,m-1);

  bool const
  pl_active = pl_vd_ratio < machine_eps;

  bool const
  vd_active = pl_vd_ratio > 1.0 / machine_eps;
      
  bool const
  eff_active = !pl_active && !vd_active;
      
  // prevent flow rule singularities if stress is zero
  ScalarT
  effective{power_law};

  if (eff_active == true) {
    effective = 1.0/((1.0 / power_law) + (1.0 / viscous_drag));
  }
  else if (vd_active == true) {
    effective = viscous_drag;
  }

  // compute slip increment
  return  g0 * effective;

}

//
// No flow rule
//
template<typename ScalarT>
ScalarT
CP::NoFlowRule<ScalarT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ScalarT const & shear,
  ScalarT const & slip_resistance)
{
  return 0.;
}

