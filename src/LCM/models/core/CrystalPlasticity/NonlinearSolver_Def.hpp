//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//
// ResidualSlipNLS
//

//
// Define nonlinear system based on residual of slip values
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::ResidualSlipNLS(
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      minitensor::Tensor<RealType, NumDimT> const & Fp_n,
      minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
      minitensor::Vector<RealType, NumSlipT> const & slip_n,
      minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt,
      CP::Verbosity verbosity)
  :
      C_(C),
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      Fp_n_(Fp_n),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_np1_(F_np1),
      dt_(dt),
      verbosity_(verbosity)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
T
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::value(
    minitensor::Vector<T, N> const & x)
{
  return Base::value(*this, x);
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, N>
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::gradient(
    minitensor::Vector<T, N> const & x)
{
  auto const
  num_unknowns = x.get_dimension();

  minitensor::Vector<T, N>
  residual(num_unknowns, minitensor::Filler::ZEROS);

  // Return immediately if something failed catastrophically.
  if (this->get_failed() == true) {
    return residual;
  }

  minitensor::Vector<T, NumSlipT>
  slip_np1(num_slip_, minitensor::Filler::ZEROS);

  minitensor::Vector<T, NumSlipT>
  state_hardening_np1(num_slip_, minitensor::Filler::ZEROS);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
    state_hardening_np1[i] = state_hardening_n_[i];
  }

  minitensor::Vector<T, NumSlipT>
  rate_slip(num_slip_, minitensor::Filler::ZEROS);

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }

  // Ensure that the slip increment will not cause overflow
  if (minitensor::norm(rate_slip * dt_) > LOG_HUGE) {
    this->set_failed("Failed on slip");
    return residual;
   }

  minitensor::Tensor<T, NumDimT>
  Lp_np1(num_dim_, minitensor::Filler::ZEROS);

  minitensor::Tensor<T, NumDimT>
  Fp_np1(num_dim_, minitensor::Filler::ZEROS);

  // Compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT, T>(
      slip_systems_,
      dt_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);

  bool
  failed{false};

  minitensor::Tensor<T, NumDimT> const
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  minitensor::Tensor4<T, NumDimT> const
  C_peeled = LCM::peel_tensor4<EvalT, T, N, CP::MAX_DIM>()(C_);

  minitensor::Tensor<T, NumDimT>
  sigma_np1(num_dim_, minitensor::Filler::ZEROS);

  minitensor::Tensor<T, NumDimT>
  S_np1(num_dim_, minitensor::Filler::ZEROS);

  minitensor::Vector<T, NumSlipT>
  shear_np1(num_slip_, minitensor::Filler::ZEROS);

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, T>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1,
      failed);

  // Ensure that the stress was calculated properly
  if (failed == true) {
    this->set_failed("Failed on stress");
    return residual;
  }

  minitensor::Vector<T, NumSlipT>
  slip_resistance(num_slip_, minitensor::Filler::ZEROS);

  // Compute state_hardening_np1
  CP::updateHardness<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_np1,
      slip_resistance,
      failed);

  // Ensure that the hardening law was calculated properly
  if (failed == true) {
    this->set_failed("Failed on hardness");
    return residual;
  }

  minitensor::Vector<T, NumSlipT>
  slip_computed(num_slip_, minitensor::Filler::ZEROS);

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_resistance,
      shear_np1,
      slip_n_,
      slip_computed,
      failed);

  // Ensure that the flow rule was calculated properly
  if (failed == true) {
    this->set_failed("Failed on flow");
    return residual;
  }

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_np1[i] - slip_computed[i];
  }

  return residual;
}

// Nonlinear system, residual based on slip increments
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Tensor<T, N>
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::hessian(
    minitensor::Vector<T, N> const & x)
{
  return Base::hessian(*this, x);
}

//
// Define nonlinear system for plastic power for slip update
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
CP::Dissipation<NumDimT, NumSlipT, EvalT>::Dissipation(
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
      minitensor::Vector<RealType, NumSlipT> const & slip_n,
      minitensor::Tensor<RealType, NumDimT> const & F_n,
      minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt,
      CP::Verbosity verbosity)
  :
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_n_(F_n),
      F_np1_(F_np1),
      dt_(dt),
      verbosity_(verbosity)
{
  num_dim_ = F_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
T
CP::Dissipation<NumDimT, NumSlipT, EvalT>::value(
    minitensor::Vector<T, N> const & x)
{
  minitensor::Vector<T, NumSlipT> const
  rate_slip_in{minitensor::Filler::ZEROS};

  minitensor::Vector<T, NumSlipT>
  state_hardening_np1{minitensor::Filler::ZEROS};

  minitensor::Vector<T, NumSlipT>
  slip_resistance{minitensor::Filler::ZEROS};

  for (int sf_index(0); sf_index < slip_families_.size(); ++sf_index)
  {
    auto const &
    slip_family = slip_families_[sf_index];

    auto
    type_hardening_law = slip_family.getHardeningLawType();

    HardeningLawFactory<NumDimT, NumSlipT> hardening_law_factory;

    auto
    phardening = hardening_law_factory.template createHardeningLaw<T>(type_hardening_law);

    phardening->harden(
      slip_family,
      slip_systems_,
      dt_,
      rate_slip_in,
      state_hardening_n_,
      state_hardening_np1,
      slip_resistance);
  }

  RealType
  dissipation{0.0};

  for (int ss_index(0); ss_index < num_slip_; ++ ss_index)
  {
    auto const &
    slip_family = slip_families_[slip_systems_.at(ss_index).slip_family_index_];

    auto
    type_flow_rule = slip_family.getFlowRuleType();

    FlowRuleFactory
    flow_rule_factory;

    auto
    pflow = flow_rule_factory.template createFlowRule<T>(type_flow_rule);

    bool
    failed;

    T const
    rate_slip = pflow->computeRateSlip(
        slip_family.pflow_parameters_,
        x[ss_index],
        slip_resistance,
        failed);

    dissipation -= x[ss_index] * rate_slip;
  }

  return dissipation;
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, N>
CP::Dissipation<NumDimT, NumSlipT, EvalT>::gradient(
    minitensor::Vector<T, N> const & x)
{
  return Base::gradient(*this, x);
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Tensor<T, N>
CP::Dissipation<NumDimT, NumSlipT, EvalT>::hessian(
    minitensor::Vector<T, N> const & x)
{
  return Base::hessian(*this, x);
}

//
// ResidualSlipHardnessNLS
//

//
// Define nonlinear system based on residual of slip and hardness values
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::ResidualSlipHardnessNLS(
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      minitensor::Tensor<RealType, NumDimT> const & Fp_n,
      minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
      minitensor::Vector<RealType, NumSlipT> const & slip_n,
      minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt,
      CP::Verbosity verbosity)
  :
      C_(C),
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      Fp_n_(Fp_n),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_np1_(F_np1),
      dt_(dt),
      verbosity_(verbosity)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
T
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::value(
    minitensor::Vector<T, N> const & x)
{
  return Base::value(*this, x);
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, N>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::gradient(
    minitensor::Vector<T, N> const & x)
{
  minitensor::Vector<T, N>
  residual(x.get_dimension(), minitensor::Filler::ZEROS);

  // Return immediately if something failed catastrophically.
  if (this->get_failed() == true) {
    return residual;
  }

  minitensor::Vector<T, NumSlipT>
  slip_np1(num_slip_, minitensor::Filler::ZEROS);

  minitensor::Vector<T, NumSlipT>
  state_hardening_np1(num_slip_, minitensor::Filler::ZEROS);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
    state_hardening_np1[i] = x[i + num_slip_];
  }

  minitensor::Vector<T, NumSlipT>
  rate_slip(num_slip_, minitensor::Filler::ZEROS);

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }

  // Ensure that the slip increment is bounded
  if (minitensor::norm_infinity(rate_slip * dt_) > LOG_HUGE) {
    this->set_failed("Failed on slip");
    return residual;
  }

  minitensor::Tensor<T, NumDimT>
  Fp_np1(num_dim_, minitensor::Filler::ZEROS);

  minitensor::Tensor<T, NumDimT>
  Lp_np1(num_dim_, minitensor::Filler::ZEROS);

  // Compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT, T>(
      slip_systems_,
      dt_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);

  bool
  failed{false};

  minitensor::Tensor<T, NumDimT> const
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  minitensor::Tensor4<T, NumDimT> const
  C_peeled = LCM::peel_tensor4<EvalT, T, N, CP::MAX_DIM>()(C_);

  minitensor::Tensor<T, NumDimT>
  sigma_np1(num_dim_, minitensor::Filler::ZEROS);

  minitensor::Tensor<T, NumDimT>
  S_np1(num_dim_, minitensor::Filler::ZEROS);

  minitensor::Vector<T, NumSlipT>
  shear_np1(num_slip_, minitensor::Filler::ZEROS);

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, T>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1,
      failed);

  // Ensure that the stress was calculated properly
  if (failed == true) {
    this->set_failed("Failed on stress");
    return residual;
  }

  minitensor::Vector<T, NumSlipT>
  state_hardening_computed(num_slip_, minitensor::Filler::ZEROS);

  minitensor::Vector<T, NumSlipT>
  slip_resistance(num_slip_, minitensor::Filler::ZEROS);

  // Compute state_hardening_np1
  state_hardening_computed = state_hardening_np1;
  CP::updateHardness<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_computed,
      slip_resistance,
      failed);

  // Ensure that the hardening law was calculated properly
  if (failed == true) {
    this->set_failed("Failed on hardness");
    return residual;
  }

  minitensor::Vector<T, NumSlipT>
  slip_computed(num_slip_, minitensor::Filler::ZEROS);

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_resistance,
      shear_np1,
      slip_n_,
      slip_computed,
      failed);

  // Ensure that the flow rule was calculated properly
  if (failed == true) {
    this->set_failed("Failed on flow");
    return residual;
  }

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_np1[i] - slip_computed[i];
    residual[i + num_slip_] =
        state_hardening_np1[i] - state_hardening_computed[i];
  }

  return residual;
}

// Nonlinear system, residual based on slip increments and hardnesses
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Tensor<T, N>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::hessian(
    minitensor::Vector<T, N> const & x)
{
  return Base::hessian(*this, x);
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
CP::ResidualSlipHardnessFN<NumDimT, NumSlipT, EvalT>::ResidualSlipHardnessFN(
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      minitensor::Tensor<RealType, NumDimT> const & Fp_n,
      minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
      minitensor::Vector<RealType, NumSlipT> const & slip_n,
      minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt,
      CP::Verbosity verbosity)
  :
      C_(C),
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      Fp_n_(Fp_n),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_np1_(F_np1),
      dt_(dt),
      verbosity_(verbosity)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
template<typename T, minitensor::Index N>
T
CP::ResidualSlipHardnessFN<NumDimT, NumSlipT, EvalT>::value(
    minitensor::Vector<T, N> const & x)
{
  // DJL
  // This is not a good design, should make calls to ResidualSlipHardnessNLS function that is almost identical.
  // Currently implemented this way for debugging only

  // Tensor mechanical state variables
  minitensor::Tensor<T, NumDimT>
  Fp_np1(num_dim_);

  minitensor::Tensor<T, NumDimT>
  Lp_np1(num_dim_);

  minitensor::Tensor<T, NumDimT>
  sigma_np1(num_dim_);

  minitensor::Tensor<T, NumDimT>
  S_np1(num_dim_);

  // Slip system state variables
  minitensor::Vector<T, NumSlipT>
  state_hardening_np1(num_slip_);

  minitensor::Vector<T, NumSlipT>
  state_hardening_computed(num_slip_);

  minitensor::Vector<T, NumSlipT>
  slip_resistance(num_slip_);

  minitensor::Vector<T, NumSlipT>
  slip_np1(num_slip_);

  minitensor::Vector<T, NumSlipT>
  slip_computed(num_slip_);

  minitensor::Vector<T, NumSlipT>
  shear_np1(num_slip_);

  minitensor::Vector<T, NumSlipT>
  rate_slip(num_slip_);

  auto const
  num_unknowns = x.get_dimension();

  minitensor::Vector<T, N>
  residual(num_unknowns);

  // Return immediately if something failed catastrophically.
  if (this->get_failed() == true) {
    residual.fill(minitensor::Filler::ZEROS);
    return 0.5 * minitensor::dot(residual, residual);
  }

  minitensor::Tensor<T, NumDimT> const
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  minitensor::Tensor4<T, NumDimT> const
  C_peeled = LCM::peel_tensor4<EvalT, T, N, CP::MAX_DIM>()(C_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
    state_hardening_np1[i] = x[i + num_slip_];
  }

  minitensor::Vector<T, NumSlipT>
  rates_slip(num_slip_, minitensor::Filler::ZEROS);

  if (dt_ > 0.0) {
    rates_slip = (slip_np1 - slip_n_) / dt_;
  }

  // Ensure that the slip increment is bounded
  if (minitensor::norm(rate_slip * dt_) > LOG_HUGE) {
    this->set_failed("Failed on slip");
    residual.fill(minitensor::Filler::ZEROS);
    return 0.5 * minitensor::dot(residual, residual);
  }

  // Compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT, T>(
      slip_systems_,
      dt_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);

  bool
  failed{false};

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, T>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1,
      failed);

  // Ensure that the stress was calculated properly
  if (failed == true) {
    this->set_failed("Failed on stress");
    residual.fill(minitensor::Filler::ZEROS);
    T val = 0.5 * minitensor::dot(residual, residual);
    return val;
  }

  // Compute state_hardening_np1
  state_hardening_computed = state_hardening_np1;
  CP::updateHardness<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_computed,
      slip_resistance,
      failed);

  // Ensure that the hardening law was calculated properly
  if (failed == true) {
    this->set_failed("Failed on hardness");
    residual.fill(minitensor::Filler::ZEROS);
    T val = 0.5 * minitensor::dot(residual, residual);
    return val;
  }

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_resistance,
      shear_np1,
      slip_n_,
      slip_computed,
      failed);

  // Ensure that the flow rule was calculated properly
  if (failed == true) {
    this->set_failed("Failed on flow");
    residual.fill(minitensor::Filler::ZEROS);
    T val = 0.5 * minitensor::dot(residual, residual);
    return val;
  }

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_np1[i] - slip_computed[i];
    residual[i + num_slip_] =
        state_hardening_np1[i] - state_hardening_computed[i];
  }

  T val = 0.5 * minitensor::dot(residual, residual);

  return val;
}
