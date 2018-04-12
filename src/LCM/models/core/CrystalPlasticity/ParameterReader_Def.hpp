//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <map>

template<typename EvalT, typename Traits>
CP::ParameterReader<EvalT, Traits>::ParameterReader(Teuchos::ParameterList *p)
  : p_(p)
{
}

template<typename EvalT, typename Traits>
CP::Verbosity
CP::ParameterReader<EvalT, Traits>::getVerbosity() const
{
  static utility::ParameterEnum<CP::Verbosity> const vmap(
    "Verbosity", CP::Verbosity::NONE,
    {
      {"None", CP::Verbosity::NONE},
      {"Low", CP::Verbosity::LOW},
      {"Medium", CP::Verbosity::MEDIUM},
      {"High", CP::Verbosity::HIGH},
      {"Debug", CP::Verbosity::DEBUG},
      {"Extreme", CP::Verbosity::EXTREME}
    });

  return vmap.get(p_);
}

template<typename EvalT, typename Traits>
CP::IntegrationScheme
CP::ParameterReader<EvalT, Traits>::getIntegrationScheme() const
{
  static utility::ParameterEnum<CP::IntegrationScheme> const imap(
    "Integration Scheme", CP::IntegrationScheme::EXPLICIT,
    {
      {"Implicit", CP::IntegrationScheme::IMPLICIT},
      {"Explicit", CP::IntegrationScheme::EXPLICIT}
    });

  return imap.get(p_);
}

template<typename EvalT, typename Traits>
CP::ResidualType
CP::ParameterReader<EvalT, Traits>::getResidualType() const
{
  static utility::ParameterEnum<CP::ResidualType> const rmap(
    "Residual Type", CP::ResidualType::SLIP,
    {
      {"Slip", CP::ResidualType::SLIP},
      {"Slip Hardness", CP::ResidualType::SLIP_HARDNESS},
      {"Constrained Slip Hardness", CP::ResidualType::CONSTRAINED_SLIP_HARDNESS}
    });

  return rmap.get(p_);
}

template<typename EvalT, typename Traits>
CP::PredictorSlip
CP::ParameterReader<EvalT, Traits>::getPredictorSlip() const
{
  static utility::ParameterEnum<CP::PredictorSlip> const pmap(
      "Slip Predictor", CP::PredictorSlip::RATE,
      {
          {"None", CP::PredictorSlip::NONE},
          {"Rate", CP::PredictorSlip::RATE},
          {"Solve", CP::PredictorSlip::SOLVE}
      });

  return pmap.get(p_);
}

template<typename EvalT, typename Traits>
minitensor::StepType
CP::ParameterReader<EvalT, Traits>::getStepType() const
{
  static utility::ParameterEnum<minitensor::StepType> const smap(
    "Nonlinear Solver Step Type", minitensor::StepType::NEWTON,
    {
      {"Newton", minitensor::StepType::NEWTON},
      {"Trust Region", minitensor::StepType::TRUST_REGION},
      {"Conjugate Gradient", minitensor::StepType::CG},
      {"Line Search Regularized", minitensor::StepType::LINE_SEARCH_REG},
      {"Newton with Line Search", minitensor::StepType::NEWTON_LS}
    });

  return smap.get(p_);
}

template<typename EvalT, typename Traits>
typename CP::ParameterReader<EvalT, Traits>::Minimizer
CP::ParameterReader<EvalT, Traits>::getMinimizer() const
{
  // TODO: This code works differently from the previous. Is this preferable?
  Minimizer
  min;

  min.rel_tol = p_->get<RealType>("Implicit Integration Relative Tolerance", 1.0e-6);
  min.abs_tol = p_->get<RealType>("Implicit Integration Absolute Tolerance", 1.0e-10);
  min.acc_tol = p_->get<RealType>("Implicit Integration Acceptable Tolerance", 1.0e-14);
  min.max_num_iter = p_->get<int>("Implicit Integration Max Iterations", 100);
  min.min_num_iter = p_->get<int>("Implicit Integration Min Iterations", 2);
  min.enforce_non_stagnation = p_->get<bool>("Implicit Integration Enforce Non-Stagnation", false);
  min.max_stagnation_iter = p_->get<int>("Implicit Integration Max Stagnation Iterations", 100);
  min.stagnation_tol = p_->get<RealType>("Implicit Integration Stagnation Tolerance", 1.0);

  return min;
}

template<typename EvalT, typename Traits>
typename CP::ParameterReader<EvalT, Traits>::RolMinimizer
CP::ParameterReader<EvalT, Traits>::getRolMinimizer() const
{
  RolMinimizer
  min;

  return min;
}

template<typename EvalT, typename Traits>
CP::SlipFamily<CP::MAX_DIM, CP::MAX_SLIP>
CP::ParameterReader<EvalT, Traits>::getSlipFamily(int index)
{
  SlipFamily<MAX_DIM, MAX_SLIP>
  slip_family;

  auto
  family_plist = p_->sublist(Albany::strint("Slip System Family", index));

  // Get flow rule parameters
  auto
  f_list = family_plist.sublist("Flow Rule");

  static utility::ParameterEnum<CP::FlowRuleType> const fmap(
    "Type", CP::FlowRuleType::UNDEFINED,
    {
      {"Power Law", CP::FlowRuleType::POWER_LAW},
      {"Thermal Activation", CP::FlowRuleType::THERMAL_ACTIVATION},
      {"Power Law with Drag", CP::FlowRuleType::POWER_LAW_DRAG}
    });

  slip_family.setFlowRuleType(fmap.get(&f_list));

  for (auto & param : slip_family.pflow_parameters_->param_map_)
  {
    auto const
    index_param = param.second;

    auto const
    value_param = f_list.get<RealType>(param.first);

    slip_family.pflow_parameters_->setParameter(index_param, value_param);
  }

  slip_family.pflow_parameters_->setTolerance();

  // set max slip increment if user specified
  RealType const max_incr = family_plist.get<RealType>("Max Slip Increment", LOG_HUGE);

  slip_family.pflow_parameters_->setMaxIncrement(max_incr);

  // Obtain hardening law parameters
  Teuchos::ParameterList
  h_list = family_plist.sublist("Hardening Law");

  static utility::ParameterEnum<CP::HardeningLawType> const hmap(
    "Type", CP::HardeningLawType::UNDEFINED,
    {
      {"Linear Minus Recovery", CP::HardeningLawType::LINEAR_MINUS_RECOVERY},
      {"Saturation", CP::HardeningLawType::SATURATION},
      {"Dislocation Density", CP::HardeningLawType::DISLOCATION_DENSITY}
    });

  slip_family.setHardeningLawType(hmap.get(&h_list));

  for (auto & param : slip_family.phardening_parameters_->param_map_)
  {
    auto const
    index_param = param.second;

    auto const
    value_param = h_list.get<RealType>(param.first);

    slip_family.phardening_parameters_->setParameter(index_param, value_param);
  }

  return slip_family;
}

