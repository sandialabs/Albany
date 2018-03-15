//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//
// Factory returning a pointer to a hardening paremeter object
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
std::shared_ptr<CP::HardeningParameterBase<NumDimT, NumSlipT>>
CP::hardeningParameterFactory(CP::HardeningLawType type_hardening_law)
{
  using HPUP = std::shared_ptr<CP::HardeningParameterBase<NumDimT, NumSlipT>>;

  switch (type_hardening_law) {

  default:
    std::cerr << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "ERROR: Unknown hardening law\n";
    exit(1);
    break;

  case HardeningLawType::LINEAR_MINUS_RECOVERY:
    return HPUP(new CP::LinearMinusRecoveryHardeningParameters<NumDimT, NumSlipT>());
    break;

  case HardeningLawType::SATURATION:
    return HPUP(new CP::SaturationHardeningParameters<NumDimT, NumSlipT>());
    break;

  case HardeningLawType::DISLOCATION_DENSITY:
    return HPUP(new CP::DislocationDensityHardeningParameters<NumDimT, NumSlipT>());
    break;

  case HardeningLawType::UNDEFINED:
    return HPUP(new CP::NoHardeningParameters<NumDimT, NumSlipT>());
    break;

  }

  return HPUP(nullptr);
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::HardeningLawFactory<NumDimT, NumSlipT>::HardeningLawFactory()
{

}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
template<typename ArgT>
utility::StaticPointer<CP::HardeningLawBase<NumDimT, NumSlipT, ArgT>>
CP::HardeningLawFactory<NumDimT, NumSlipT>::createHardeningLaw(
    HardeningLawType type_hardening_law) const
{
  switch (type_hardening_law) {

    default:
      std::cerr << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "ERROR: Unknown hardening law\n";
      exit(1);
      break;

    case HardeningLawType::LINEAR_MINUS_RECOVERY:
      return allocator_.create<LinearMinusRecoveryHardeningLaw<NumDimT, NumSlipT, ArgT>>();
      break;

    case HardeningLawType::SATURATION:
      return allocator_.create<SaturationHardeningLaw<NumDimT, NumSlipT, ArgT>>();
      break;

    case HardeningLawType::DISLOCATION_DENSITY:
      return allocator_.create<DislocationDensityHardeningLaw<NumDimT, NumSlipT, ArgT>>();
      break;

    case HardeningLawType::UNDEFINED:
      return allocator_.create<NoHardeningLaw<NumDimT, NumSlipT, ArgT>>();
      break;
  }

  return nullptr;
}

//
// Linear hardening with recovery
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::LinearMinusRecoveryHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(minitensor::Filler::ONES);

  return;
}

//
//
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::LinearMinusRecoveryHardeningLaw<NumDimT, NumSlipT, ArgT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  minitensor::Vector<ArgT, NumSlipT> const & rate_slip,
  minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
  minitensor::Vector<ArgT, NumSlipT> & state_hardening_np1,
  minitensor::Vector<ArgT, NumSlipT> & slip_resistance,
  bool & failed)
{
  using Params = LinearMinusRecoveryHardeningParameters<NumDimT, NumSlipT>;

  minitensor::Index const
  num_slip_sys = slip_family.num_slip_sys_;

  minitensor::Vector<ArgT, NumSlipT>
  rate_slip_abs(num_slip_sys);

  for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_family.slip_system_indices_[ss_index];

    rate_slip_abs[ss_index] = std::fabs(rate_slip[ss_index_global]);
  }

  minitensor::Vector<ArgT, NumSlipT> const
  driver_hardening = slip_family.latent_matrix_ * rate_slip_abs;

  auto const &
  phardening_params = slip_family.phardening_parameters_;

  auto const
  modulus_recovery = phardening_params->getParameter(Params::MODULUS_RECOVERY);

  auto const
  modulus_hardening = phardening_params->getParameter(Params::MODULUS_HARDENING);

  auto const
  hardness_initial = phardening_params->getParameter(
      Params::STATE_HARDENING_INITIAL);

  if (modulus_recovery > 0.0)
  {
    for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
    {
      auto const
      ss_index_global = slip_family.slip_system_indices_(ss_index);

      // FIXME: there is no guard against log(x), x<0
      RealType const
      effective_slip_n = -1.0 / modulus_recovery *
        std::log(1.0 - modulus_recovery / modulus_hardening * (state_hardening_n[ss_index_global] - hardness_initial));

      state_hardening_np1[ss_index_global] = modulus_hardening / modulus_recovery * (1.0 -
        std::exp(-modulus_recovery * (effective_slip_n + dt * driver_hardening[ss_index]))) + hardness_initial;

      slip_resistance[ss_index_global] = state_hardening_np1[ss_index_global];
    }
  }
  else
  {
    for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
    {
      auto const
      ss_index_global = slip_family.slip_system_indices_(ss_index);

      state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global] +
          modulus_hardening * dt * driver_hardening[ss_index];

      slip_resistance[ss_index_global] = state_hardening_np1[ss_index_global];
    }
  }

  return;
}

//
// Saturation hardening
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::SaturationHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(minitensor::Filler::ZEROS);

  for (minitensor::Index ss_index_i(0); ss_index_i < slip_family.num_slip_sys_; ++ss_index_i) {

    auto const
    slip_system_i = slip_systems[slip_family.slip_system_indices_[ss_index_i]];

    for (minitensor::Index ss_index_j(0); ss_index_j < slip_family.num_slip_sys_; ++ss_index_j) {

      auto const
      slip_system_j = slip_systems[slip_family.slip_system_indices_[ss_index_j]];

      slip_family.latent_matrix_(ss_index_i, ss_index_j) =
        std::fabs(minitensor::dotdot(
          minitensor::sym(slip_system_i.projector_),
          minitensor::sym(slip_system_j.projector_)));
    }
  }

  return;
}

//
//
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::SaturationHardeningLaw<NumDimT, NumSlipT, ArgT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  minitensor::Vector<ArgT, NumSlipT> const & rate_slip,
  minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
  minitensor::Vector<ArgT, NumSlipT> & state_hardening_np1,
  minitensor::Vector<ArgT, NumSlipT> & slip_resistance,
  bool & failed)
{
  using Params = SaturationHardeningParameters<NumDimT, NumSlipT>;

  minitensor::Index const
  num_slip_sys = slip_family.num_slip_sys_;

  minitensor::Vector<ArgT, NumSlipT>
  rate_slip_abs(num_slip_sys);
  rate_slip_abs.fill(minitensor::Filler::ZEROS);

  for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_family.slip_system_indices_(ss_index);

    auto const &
    slip_rate = rate_slip(ss_index_global);

    rate_slip_abs(ss_index) = std::fabs(slip_rate);
  }

  minitensor::Vector<ArgT, NumSlipT> const
  driver_hardening = 2.0 * slip_family.latent_matrix_ * rate_slip_abs;

  ArgT
  effective_slip_rate{minitensor::norm_1(rate_slip_abs)};

  if (effective_slip_rate < CP::TINY)
  {
    for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
    {
      auto const &
      ss_index_global = slip_family.slip_system_indices_[ss_index];

      state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global]
          + dt * driver_hardening[ss_index];

      slip_resistance[ss_index_global] = state_hardening_np1[ss_index_global];
    }
    return;
  }

  auto const
  phardening_params = slip_family.phardening_parameters_;

  auto const
  param_map = phardening_params->param_map_;

  auto const
  stress_saturation_initial = phardening_params->getParameter(
      Params::STRESS_SATURATION_INITIAL);

  auto const
  rate_slip_reference = phardening_params->getParameter(
      Params::RATE_SLIP_REFERENCE);

  auto const
  exponent_saturation = phardening_params->getParameter(
      Params::EXPONENT_SATURATION);

  auto const
  rate_hardening = phardening_params->getParameter(Params::RATE_HARDENING);

  auto const
  resistance_slip_initial = phardening_params->getParameter(
      Params::STATE_HARDENING_INITIAL);

  RealType const
  min_tol = phardening_params->min_tol_;

  RealType const
  max_tol = phardening_params->max_tol_;

  for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const &
    ss_index_global = slip_family.slip_system_indices_[ss_index];

    ArgT
    stress_saturation{stress_saturation_initial};

    ArgT const
    ratio_rate = effective_slip_rate / rate_slip_reference;

    if (exponent_saturation > 0.0) {

      if (ratio_rate > max_tol) {
        failed = true;
        return;
      }

      stress_saturation = stress_saturation_initial * std::pow(
          ratio_rate, exponent_saturation);
    }

    ArgT &
    state_hard = state_hardening_np1[ss_index_global];

    slip_resistance[ss_index_global] = state_hard;

    ArgT
    ratio_hardening{0.0};

    // if (state_hardening_np1[ss_index_global] < stress_saturation)
    // {
      ratio_hardening = (stress_saturation - state_hard) /
        (stress_saturation - resistance_slip_initial);
    // }

    // if (driver_hardening[ss_index] !=0 )
    // {
    state_hard = state_hardening_n[ss_index_global] +
      dt * rate_hardening * driver_hardening[ss_index] * ratio_hardening;

    // }
    // else
    // {
    //   state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global];
    // }
  }

  return;
}

//
// Dislocation-density based hardening
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::DislocationDensityHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  minitensor::Index const
  num_dim = slip_systems[0].s_.get_dimension();

  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(minitensor::Filler::ZEROS);

  for (minitensor::Index ss_index_i(0); ss_index_i < slip_family.num_slip_sys_; ++ss_index_i)
  {
    auto const &
    slip_system_i = slip_systems[slip_family.slip_system_indices_[ss_index_i]];

    minitensor::Vector<RealType, CP::MAX_DIM>
    normal_i(num_dim);

    normal_i = slip_system_i.n_;

    for (minitensor::Index ss_index_j(0); ss_index_j < slip_family.num_slip_sys_; ++ss_index_j)
    {
      auto const &
      slip_system_j = slip_systems[slip_family.slip_system_indices_[ss_index_j]];

      minitensor::Vector<RealType, CP::MAX_DIM>
      direction_j = slip_system_j.s_;

      minitensor::Vector<RealType, CP::MAX_DIM>
      normal_j = slip_system_j.n_;

      minitensor::Vector<RealType, CP::MAX_DIM>
      transverse_j = minitensor::unit(minitensor::cross(normal_j, direction_j));

      slip_family.latent_matrix_(ss_index_i, ss_index_j) =
          std::abs(minitensor::dot(normal_i, transverse_j));
    }
  }

  slip_family.aux_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.aux_matrix_.fill(minitensor::Filler::ZEROS);

  for (minitensor::Index ss_index_i(0); ss_index_i < slip_family.num_slip_sys_; ++ss_index_i)
  {
    for (minitensor::Index ss_index_j(0); ss_index_j < slip_family.num_slip_sys_; ++ss_index_j)
    {
      slip_family.aux_matrix_(ss_index_i, ss_index_j) =
        std::sqrt(1.0 - std::pow(slip_family.latent_matrix_(ss_index_i, ss_index_j), 2));
    }
  }

  return;
}

//
//
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::DislocationDensityHardeningLaw<NumDimT, NumSlipT, ArgT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  minitensor::Vector<ArgT, NumSlipT> const & rate_slip,
  minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
  minitensor::Vector<ArgT, NumSlipT> & state_hardening_np1,
  minitensor::Vector<ArgT, NumSlipT> & slip_resistance,
  bool & failed)
{
  using Params = DislocationDensityHardeningParameters<NumDimT, NumSlipT>;

  minitensor::Index const
  num_slip_sys = slip_family.num_slip_sys_;

  //
  // Compute the effective dislocation density at step n
  //
  bool dd_negative = false;
  for (minitensor::Index ss_index_i(0); ss_index_i < num_slip_sys; ++ss_index_i)
  {
    if (state_hardening_np1[ss_index_i] < 0.0)
    {
      dd_negative = true;
      // std::cout << state_hardening_np1;
    }
  }
  if (dd_negative)
  {
      std::cout << "Warning: Dislocation density at np1 is negative" << std::endl;
  }

  minitensor::Vector<ArgT, NumSlipT>
  densities_forest = slip_family.latent_matrix_ * state_hardening_np1;

  //
  // Update dislocation densities
  //
  auto const
  phardening_params = slip_family.phardening_parameters_;

  auto const
  param_map = phardening_params->param_map_;

  RealType const
  factor_generation = phardening_params->getParameter(Params::FACTOR_GENERATION);

  RealType const
  factor_annihilation = phardening_params->getParameter(Params::FACTOR_ANNIHILATION);

  RealType const
  factor_geometry_dislocation = phardening_params->getParameter(
      Params::FACTOR_GEOMETRY_DISLOCATION);

  RealType const
  modulus_shear = phardening_params->getParameter(Params::MODULUS_SHEAR);

  RealType const
  magnitude_burgers = phardening_params->getParameter(Params::MAGNITUDE_BURGERS);

  for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_family.slip_system_indices_[ss_index];

    if (densities_forest[ss_index] < 0.0) {
      failed = true;
      return;
    }

    ArgT const
    generation = factor_generation * std::sqrt(densities_forest[ss_index]);

    ArgT const
    annihilation = factor_annihilation * state_hardening_np1[ss_index_global];

    ArgT
    driver_hardening;

    if (generation > annihilation)
    {
      driver_hardening = generation - annihilation;
    }
    else
    {
      // DJL this probably causes problems with AD types, sets derivative information to zero
      driver_hardening = 0.0;
    }

    state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global] +
      dt * driver_hardening * std::abs(rate_slip[ss_index_global]);
  }

  minitensor::Vector<ArgT, NumSlipT> const
  densities_parallel = slip_family.aux_matrix_ * state_hardening_np1;

  for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {

    auto const
    ss_index_global = slip_family.slip_system_indices_[ss_index];

    if (densities_parallel[ss_index] < 0.0) {
      failed = true;
      return;
    }

    // Compute the slip resistance
    slip_resistance[ss_index_global] =
        factor_geometry_dislocation * modulus_shear * magnitude_burgers *
        std::sqrt(densities_parallel[ss_index]);
  }
}

//
// No hardening
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::NoHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(minitensor::Filler::ZEROS);

  return;
}

//
//
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::NoHardeningLaw<NumDimT, NumSlipT, ArgT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family,
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  minitensor::Vector<ArgT, NumSlipT> const & rate_slip,
  minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
  minitensor::Vector<ArgT, NumSlipT> & state_hardening_np1,
  minitensor::Vector<ArgT, NumSlipT> & slip_resistance,
  bool & failed)
{
  minitensor::Index const
  num_slip_sys = slip_family.num_slip_sys_;

  auto const
  slip_system_indices = slip_family.slip_system_indices_;

  for (minitensor::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_system_indices[ss_index];

    state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global];
    slip_resistance[ss_index_global] = state_hardening_np1[ss_index_global];
  }
}
