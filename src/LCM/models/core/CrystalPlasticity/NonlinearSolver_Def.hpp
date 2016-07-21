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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::ResidualSlipNLS(
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C),
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      Fp_n_(Fp_n),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_np1_(F_np1),
      dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
T
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
  ResidualSlipNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::value(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid2::Vector<T, N> const & x)
{
  // Get a convenience reference to the failed flag in case it is used more
  // than once.
  bool &
  failed = Intrepid2::Function_Base<
  ResidualSlipNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::failed;

  // Tensor mechanical state variables
  Intrepid2::Tensor<T, NumDimT>
  Fp_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  Lp_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  sigma_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  S_np1(num_dim_);

  // Slip system state variables
  Intrepid2::Vector<T, NumSlipT>
  state_hardening_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_resistance(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_computed(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  shear_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  rate_slip(num_slip_);

  auto const
  num_unknowns = x.get_dimension();

  Intrepid2::Vector<T, N>
  residual(num_unknowns);

  // Return immediately if something failed catastrophically.
  if (failed == true) {
    residual.fill(Intrepid2::ZEROS);
    return residual;
  }

  Intrepid2::Tensor<T, NumDimT> const
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  Intrepid2::Tensor4<T, NumDimT> const
  C_peeled = LCM::peel_tensor4<EvalT, T, N, NumDimT>()(C_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
  }

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rate_slip.fill(Intrepid2::ZEROS);
  }

  // Ensure that the slip increment is bounded
   if (Intrepid2::norm(rate_slip * dt_) > 1.0) {
       failed =  true;
       return residual;
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

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, T, T>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1);

  // Compute state_hardening_np1
  CP::updateHardness<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_np1,
      slip_resistance);

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_resistance,
      shear_np1,
      slip_n_,
      slip_computed);

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_np1[i] - slip_computed[i];
  }

  return residual;
}

// Nonlinear system, residual based on slip increments
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Tensor<T, N>
CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>::hessian(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
      ResidualSlipNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::hessian(
      *this,
      x);
}


//
// ResidualSlipHardnessNLS
//

//
// Define nonlinear system based on residual of slip and hardness values
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::ResidualSlipHardnessNLS(
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C),
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      Fp_n_(Fp_n),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_np1_(F_np1),
      dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
T
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
  ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::value(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid2::Vector<T, N> const & x)
{
  // Get a convenience reference to the failed flag in case it is used more
  // than once.
  bool &
  failed = Intrepid2::Function_Base<
  ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::failed;

  // Tensor mechanical state variables
  Intrepid2::Tensor<T, NumDimT>
  Fp_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  Lp_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  sigma_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  S_np1(num_dim_);

  // Slip system state variables
  Intrepid2::Vector<T, NumSlipT>
  state_hardening_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  state_hardening_computed(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_resistance(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_computed(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  shear_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_residual(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  rate_slip(num_slip_);

  auto const
  num_unknowns = x.get_dimension();

  Intrepid2::Vector<T, N>
  residual(num_unknowns);

  // Return immediately if something failed catastrophically.
  if (failed == true) {
    residual.fill(Intrepid2::ZEROS);
    return residual;
  }

  Intrepid2::Tensor<T, NumDimT> const
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  Intrepid2::Tensor4<T, NumDimT> const
  C_peeled = LCM::peel_tensor4<EvalT, T, N, NumDimT>()(C_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
    state_hardening_np1[i] = x[i + num_slip_];
  }

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rate_slip.fill(Intrepid2::ZEROS);
  }

  // Ensure that the slip increment is bounded
  if (Intrepid2::norm(rate_slip * dt_) > 1.0) {
    failed =  true;
    return residual;
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
  
  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, T, T>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1);

  // Compute state_hardening_np1
  CP::updateHardness<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_computed,
      slip_resistance);

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT, T>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_resistance,
      shear_np1,
      slip_n_,
      slip_computed);

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_np1[i] - slip_computed[i];
    residual[i + num_slip_] = 
        state_hardening_np1[i] - state_hardening_computed[i];
  }
  
  return residual;
}

// Nonlinear system, residual based on slip increments and hardnesses
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Tensor<T, N>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::hessian(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
      ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::hessian(
      *this,
      x);
}

//
// ExplicitUpdateNLS
//

//
// Define nonlinear system for explicit update
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>::ExplicitUpdateNLS(
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C),
      slip_systems_(slip_systems),
      slip_families_(slip_families),
      Fp_n_(Fp_n),
      state_hardening_n_(state_hardening_n),
      slip_n_(slip_n),
      F_np1_(F_np1),
      dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = state_hardening_n_.get_dimension();
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
T
CP::ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
  ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::value(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
CP::ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid2::Vector<T, N> const & x)
{
  // Get a convenience reference to the failed flag in case it is used more
  // than once.
  bool &
  failed = Intrepid2::Function_Base<
  ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::failed;

  // Tensor mechanical state variables
  Intrepid2::Tensor<T, NumDimT>
  Fp_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  Lp_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  sigma_np1(num_dim_);

  Intrepid2::Tensor<T, NumDimT>
  S_np1(num_dim_);

  // Slip system state variables
  Intrepid2::Vector<T, NumSlipT>
  state_hardening_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_resistance(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  slip_computed(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  shear_np1(num_slip_);

  Intrepid2::Vector<T, NumSlipT>
  rate_slip(num_slip_);

  auto const
  num_unknowns = x.get_dimension();

  Intrepid2::Vector<T, N>
  residual(num_unknowns);

  // Return immediately if something failed catastrophically.
  if (failed == true) {
    residual.fill(Intrepid2::ZEROS);
    return residual;
  }

  Intrepid2::Tensor<T, NumDimT> const
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  Intrepid2::Tensor4<T, NumDimT> const
  C_peeled = LCM::peel_tensor4<EvalT, T, N, NumDimT>()(C_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
  }

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rate_slip.fill(Intrepid2::ZEROS);
  }

  // Ensure that the slip increment is bounded
   if (Intrepid2::norm(rate_slip * dt_) > 1.0) {
       failed =  true;
       return residual;
   }

  Intrepid2::Tensor<T, CP::MAX_DIM>
  Fp_n_FAD(num_dim_);

  for (auto i = 0; i < Fp_n_.get_number_components(); ++i) {
    Fp_n_FAD[i] = Fp_n_[i];
  }

  // compute sigma_np1, S_np1, and shear_np1 using Fp_n
  CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, T, T>(
    slip_systems_, 
    C_peeled, 
    F_np1_peeled, 
    Fp_n_FAD, 
    sigma_np1, 
    S_np1, 
    shear_np1);

  // compute state_hardening_np1 using slip_n
  CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, T>(
    slip_systems_, 
    slip_families_,
    dt_,
    rate_slip, 
    state_hardening_n_, 
    state_hardening_np1,
    slip_resistance);

  // compute slip_np1
  CP::updateSlip<CP::MAX_DIM, CP::MAX_SLIP, T>(
    slip_systems_,
    slip_families_,
    dt_,
    slip_resistance,
    shear_np1,
    slip_n_,
    slip_np1);

  // compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP, T>(
    slip_systems_, 
    dt_,
    slip_n_, 
    slip_np1, 
    Fp_n_, 
    Lp_np1, 
    Fp_np1);

  // compute sigma_np1, S_np1, and shear_np1 using Fp_np1
  CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, T, T>(
    slip_systems_, 
    C_peeled, 
    F_np1_peeled, 
    Fp_np1, 
    sigma_np1, 
    S_np1, 
    shear_np1);

  // compute slip_np1
  CP::updateSlip<CP::MAX_DIM, CP::MAX_SLIP, T>(
    slip_systems_,
    slip_families_,
    dt_,
    slip_resistance,
    shear_np1,
    slip_n_,
    slip_computed);

  for (int i = 0; i< num_slip_; ++i) {
    residual[i] = slip_np1[i] - slip_computed[i];
  }

  return residual;
}

// Hessian of explicit update NLS
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Tensor<T, N>
CP::ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>::hessian(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
      ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::hessian(
      *this,
      x);
}


