//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>::ExplicitIntegrator(
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : Base(slip_systems, slip_families, plasticity_state, slip_state, C, F_np1, dt)
{

}

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>::update(
    Intrepid2::Vector<ScalarT, NumSlipT> & residual) const
{
  Intrepid2::Tensor<ScalarT, NumDimT>
  Fp_n_FAD(plasticity_state_.Fp_n_.get_dimension());

  for (auto i = 0; i < plasticity_state_.Fp_n_.get_number_components(); ++i) {
    Fp_n_FAD[i] = plasticity_state_.Fp_n_[i];
  }

  Intrepid2::Vector<ScalarT, NumSlipT>
  slip_rate_FAD(slip_state_.rate_.get_dimension());

  for (auto i = 0; i < slip_state_.rate_.get_number_components(); ++i) {
    slip_rate_FAD[i] = slip_state_.rate_[i];
  }

  // compute sigma_np1, S_np1, and shear_np1 using Fp_n
  CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, ScalarT, ScalarT>(
    slip_systems_, 
    C_, 
    F_np1_, 
    Fp_n_FAD, 
    plasticity_state_.sigma_np1_, 
    plasticity_state_.S_np1_, 
    slip_state_.shear_np1_);

  // compute state_hardening_np1 using slip_n
  CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    slip_rate_FAD,
    slip_state_.hardening_n_, 
    slip_state_.hardening_np1_,
    slip_state_.resistance_);

  // compute slip_np1
  CP::updateSlip<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    slip_state_.resistance_,
    slip_state_.shear_np1_,
    slip_state_.slip_n_,
    slip_state_.slip_np1_);

  // compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
    slip_systems_, 
    dt_,
    slip_state_.slip_n_,
    slip_state_.slip_np1_,
    plasticity_state_.Fp_n_,
    plasticity_state_.Lp_np1_, 
    plasticity_state_.Fp_np1_);

  // compute sigma_np1, S_np1, and shear_np1 using Fp_np1
  CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, ScalarT, ScalarT>(
    slip_systems_, 
    C_, 
    F_np1_, 
    plasticity_state_.Fp_np1_, 
    plasticity_state_.sigma_np1_, 
    plasticity_state_.S_np1_, 
    slip_state_.shear_np1_);

  Intrepid2::Vector<ScalarT, NumSlipT>
  slip_computed(slip_state_.slip_np1_.get_dimension());

  // compute slip_np1
  CP::updateSlip<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    slip_state_.resistance_,
    slip_state_.shear_np1_,
    slip_state_.slip_n_,
    slip_computed);

  residual = slip_state_.slip_np1_ - slip_computed;
}


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitSlipIntegrator(
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : Base(slip_systems, slip_families, plasticity_state, slip_state, C, F_np1, dt)
{

}


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>::update(
    Intrepid2::Vector<ScalarT, NumSlipT> & residual) const
{
#if 0
  auto num_slip = slip_state_.slip_n_.get_dimension();

  // Unknown for solver
  Intrepid2::Vector<ScalarT, NumSlipT>
  x(num_slip);

  using NonlinearSolver = CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>; 

  NonlinearSolver nls(C_, slip_systems_, slip_families_, plasticity_state_.Fp_n_,
                      slip_state_.hardening_n_, slip_state_.slip_n_,
                      F_np1_, dt_);

  using StepType = Intrepid2::StepBase<NonlinearSolver, ValueT, NumSlipT>;

  std::unique_ptr<StepType>
  pstep = Intrepid2::stepFactory<NonlinearSolver, ValueT, NumSlipT>(step_type_);
  
  // Initial guess for x should be slip_np1
  for (int i = 0; i < num_slip; ++i) {
    x(i) = Sacado::ScalarValue<ScalarT>::eval(slip_state_.slip_np1_(i));
  }

  LCM::MiniSolver<Minimizer, StepType, NonlinearSolver, EvalT, NumSlipT>
  mini_solver(minimizer_, *pstrep, nls, x);

  // Write slip back out from x
  for (int i = 0; i < num_slip; ++i) {
    slip_state_.slip_np1_(i) = x(i);
  }

  // Compute slip rate
  slip_state_.rate_.fill(Intrepid2::ZEROS);
  if (dt_ > 0.0) {
    slip_state_.rate_ = (slip_state_.slip_np1_ - slip_state_.slip_n_) / dt_;
  }

  // Update hardness
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_state_.rate_,
      slip_state_.hardening_n_,
      slip_state_.hardening_np1_,
      slip_state_.resistance_);
#endif
}



