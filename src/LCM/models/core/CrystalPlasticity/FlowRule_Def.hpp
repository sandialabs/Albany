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
// Factory returning a pointer to a flow rule object
//
template<typename ArgT>
utility::StaticPointer<CP::FlowRuleBase<ArgT>>
CP::FlowRuleFactory::createFlowRule(FlowRuleType type_flow_rule) const
{
  switch (type_flow_rule) {
    
    default:
      std::cerr << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "ERROR: Unknown flow rule\n";
      exit(1);
      break;

    case FlowRuleType::POWER_LAW:
      return allocator_.create<PowerLawFlowRule<ArgT>>();
      break;

    case FlowRuleType::POWER_LAW_DRAG:
      return allocator_.create<PowerLawDragFlowRule<ArgT>>();
      break;

    case FlowRuleType::THERMAL_ACTIVATION:
      return allocator_.create<ThermalActivationFlowRule<ArgT>>();
      break;

    case FlowRuleType::UNDEFINED:
      return allocator_.create<NoFlowRule<ArgT>>();
      break;
  }

  return nullptr;
}


//
// Power law flow rule
//
template<typename ArgT>
ArgT
CP::PowerLawFlowRule<ArgT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ArgT const & shear,
  ArgT const & slip_resistance)
{
  using Params = PowerLawFlowParameters;

  // Material properties
  RealType const
  m = pflow_parameters->getParameter(Params::EXPONENT_RATE);

  RealType const
  g0 = pflow_parameters->getParameter(Params::RATE_SLIP_REFERENCE);

  RealType const
  min_tol = pflow_parameters->min_tol_;

  ArgT const
  ratio_stress = shear / slip_resistance;

  bool const
  finite_rate = std::fabs(ratio_stress) > min_tol;

  // carry derivative info from ratio_stress
  ArgT
  rate_slip{0.0 * ratio_stress};

  if (finite_rate == true) {
    // Compute slip increment
    rate_slip = g0 * std::pow(std::fabs(ratio_stress), m - 1) * ratio_stress;
  }

  return rate_slip;
}

//
// Thermally-activated flow rule
//
template<typename ArgT>
ArgT
CP::ThermalActivationFlowRule<ArgT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ArgT const & shear,
  ArgT const & slip_resistance)
{
  using Params = ThermalActivationFlowParameters;

  //
  // Material properties
  //
  RealType const
  g0 = pflow_parameters->getParameter(Params::RATE_SLIP_REFERENCE);

  RealType const
  F0 = pflow_parameters->getParameter(Params::ENERGY_ACTIVATION);

  RealType const
  s_t = pflow_parameters->getParameter(Params::RESISTANCE_THERMAL);

  RealType const
  p = pflow_parameters->getParameter(Params::EXPONENT_P);

  RealType const
  q = pflow_parameters->getParameter(Params::EXPONENT_Q);

  RealType const
  min_tol = pflow_parameters->min_tol_;

  ArgT const
  ratio_stress = std::max(0.0, (std::fabs(shear) - slip_resistance) / s_t);

  // carry derivative info from ratio_stress
  ArgT
  rate_slip{0.0 * ratio_stress};

  // Compute slip increment
  if (ratio_stress > min_tol) {
    RealType const
    sign = shear < 0.0 ? -1.0 : 1.0;

    rate_slip = 
      g0 * std::exp(-F0 * std::pow(1.0 - std::pow(ratio_stress, p), q)) * sign;
  }

  return rate_slip;
}

//
// Power law with Drag flow rule
//
template<typename ArgT>
ArgT
CP::PowerLawDragFlowRule<ArgT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ArgT const & shear,
  ArgT const & slip_resistance)
{     
  using Params = PowerLawDragFlowParameters;

  // Material properties
  RealType const
  m = pflow_parameters->getParameter(Params::EXPONENT_RATE);

  RealType const
  g0 = pflow_parameters->getParameter(Params::RATE_SLIP_REFERENCE);

  RealType const
  coefficient_drag = pflow_parameters->getParameter(Params::COEFFICIENT_DRAG);

  RealType const
  min_tol = pflow_parameters->min_tol_;

  ArgT const
  ratio_stress = shear / slip_resistance;

  // Compute drag term
  ArgT const
  viscous_drag = ratio_stress / coefficient_drag;

  bool const
  finite_power_law = std::fabs(ratio_stress) > min_tol;

  // carry derivative info from ratio_stress
  ArgT
  power_law{0.0 * ratio_stress};

  if (finite_power_law == true) {
    power_law = std::pow(std::fabs(ratio_stress), m - 1) * ratio_stress;
  }

  ArgT
  pl_vd_ratio = coefficient_drag * std::pow(std::fabs(ratio_stress), m - 1);

  bool const
  pl_active = pl_vd_ratio < CP::MACHINE_EPS;

  bool const
  vd_active = pl_vd_ratio > 1.0 / CP::MACHINE_EPS;
      
  bool const
  eff_active = !pl_active && !vd_active;
      
  // prevent flow rule singularities if stress is zero
  ArgT
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
template<typename ArgT>
ArgT
CP::NoFlowRule<ArgT>::
computeRateSlip(
  std::shared_ptr<CP::FlowParameterBase> const & pflow_parameters,
  ArgT const & shear,
  ArgT const & slip_resistance)
{
  return 0.;
}

