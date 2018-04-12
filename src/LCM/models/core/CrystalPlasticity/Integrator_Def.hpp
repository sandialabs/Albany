//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::Integrator<EvalT, NumDimT, NumSlipT>::forceGlobalLoadStepReduction(
  std::string const & message) const
{
  ALBANY_ASSERT(nox_status_test_.is_null() == false, "Invalid NOX status test");
  nox_status_test_->status_ = NOX::StatusTest::Failed;
  nox_status_test_->status_message_ = message;
}

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::Integrator<EvalT, NumDimT, NumSlipT>::setWarningStatus(
  std::string const & message) const
{
  // only output warning messages if desired
  if (verbosity_ >= CP::Verbosity::LOW) {
    std::cout << "Warning: " << message << std::endl;
  }
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::IntegratorFactory<EvalT, NumDimT, NumSlipT>::IntegratorFactory(
      utility::StaticAllocator & allocator,
      Minimizer const & minimizer,
      RolMinimizer const & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      CP::StateMechanical<ScalarT, NumDimT> & state_mechanical,
      CP::StateInternal<ScalarT, NumSlipT> & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity verbosity)
  : allocator_(allocator),
    minimizer_(minimizer),
    rol_minimizer_(rol_minimizer),
    step_type_(step_type),
    nox_status_test_(nox_status_test),
    slip_systems_(slip_systems),
    slip_families_(slip_families),
    state_mechanical_(state_mechanical),
    state_internal_(state_internal),
    C_(C),
    dt_(dt),
    verbosity_(verbosity)
{}

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
utility::StaticPointer<typename CP::IntegratorFactory<EvalT, NumDimT, NumSlipT>::IntegratorBase>
CP::IntegratorFactory<EvalT, NumDimT, NumSlipT>::operator()(
    CP::IntegrationScheme integration_scheme,
    CP::ResidualType residual_type) const
{
  if(verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << ">>> in IntegratorFactory::operator\n";
  }

  switch (integration_scheme)
  {
    case CP::IntegrationScheme::EXPLICIT:
    {
      using IntegratorType = CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>;
      return allocator_.create<IntegratorType>(
          nox_status_test_,
          slip_systems_,
          slip_families_,
          state_mechanical_,
          state_internal_,
          C_,
          dt_,
          verbosity_);

    } break;

    case CP::IntegrationScheme::IMPLICIT:
    {
      switch (residual_type)
      {
        case CP::ResidualType::SLIP:
        {
          using IntegratorType
            = CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>;
          return allocator_.create<IntegratorType>(
              minimizer_,
              rol_minimizer_,
              step_type_,
              nox_status_test_,
              slip_systems_,
              slip_families_,
              state_mechanical_,
              state_internal_,
              C_,
              dt_,
              verbosity_);
        } break;

        case CP::ResidualType::SLIP_HARDNESS:
        {
          using IntegratorType
            = CP::ImplicitSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>;
          return allocator_.create<IntegratorType>(
              minimizer_,
              rol_minimizer_,
              step_type_,
              nox_status_test_,
              slip_systems_,
              slip_families_,
              state_mechanical_,
              state_internal_,
              C_,
              dt_,
              verbosity_);
        } break;

        case CP::ResidualType::CONSTRAINED_SLIP_HARDNESS:
        {
          using IntegratorType
            = CP::ImplicitConstrainedSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>;
          return allocator_.create<IntegratorType>(
              minimizer_,
              rol_minimizer_,
              step_type_,
              nox_status_test_,
              slip_systems_,
              slip_families_,
              state_mechanical_,
              state_internal_,
              C_,
              dt_,
              verbosity_);
        } break;

        default:
        {
          // throw
          return nullptr;
        } break;
      }
    } break;

    default:
    {
      return nullptr;
      // throw
    } break;
  }

}

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>::ExplicitIntegrator(
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity)
  : Base(
      nox_status_test,
      slip_systems,
      slip_families,
      state_mechanical,
      state_internal,
      C,
      dt,
      verbosity)
{

}

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>::update() const
{
  minitensor::Vector<ScalarT, NumSlipT>
  residual(this->num_slip_);

  minitensor::Tensor<ScalarT, NumDimT>
  Fp_n_FAD(state_mechanical_.Fp_n_.get_dimension());

  for (minitensor::Index i = 0; i < state_mechanical_.Fp_n_.get_number_components(); ++i) {
    Fp_n_FAD[i] = state_mechanical_.Fp_n_[i];
  }

  bool
  failed{false};

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "F_{n+1}" << std::endl;
    std::cout << state_mechanical_.F_np1_ << std::endl;
  }

  // compute sigma_np1, S_np1, and shear_np1 using Fp_n
  CP::computeStress<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    C_,
    state_mechanical_.F_np1_,
    Fp_n_FAD,
    state_mechanical_.sigma_np1_,
    state_mechanical_.S_np1_,
    state_internal_.shear_np1_,
    failed);

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "S_{n+1}" << std::endl;
    std::cout << state_mechanical_.S_np1_ << std::endl;
  }

  // compute state_hardening_np1 using slip_n
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    state_internal_.rates_slip_,
    state_internal_.hardening_n_,
    state_internal_.hardening_np1_,
    state_internal_.resistance_,
    failed);

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "resistance_{n+1}" << std::endl;
    std::cout << state_internal_.resistance_ << std::endl;
  }

  // compute slip_np1
  CP::updateSlip<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    state_internal_.resistance_,
    state_internal_.shear_np1_,
    state_internal_.slip_n_,
    state_internal_.slip_np1_,
    failed);

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "slip_{n+1}" << std::endl;
    std::cout << state_internal_.slip_np1_ << std::endl;
  }

  // compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    dt_,
    state_internal_.slip_n_,
    state_internal_.slip_np1_,
    state_mechanical_.Fp_n_,
    state_mechanical_.Lp_np1_,
    state_mechanical_.Fp_np1_);

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "Fp_{n+1}" << std::endl;
    std::cout << state_mechanical_.Fp_np1_ << std::endl;
  }

  // compute sigma_np1, S_np1, and shear_np1 using Fp_np1
  CP::computeStress<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    C_,
    state_mechanical_.F_np1_,
    state_mechanical_.Fp_np1_,
    state_mechanical_.sigma_np1_,
    state_mechanical_.S_np1_,
    state_internal_.shear_np1_,
    failed);

  minitensor::Vector<ScalarT, NumSlipT>
  slip_computed(state_internal_.slip_np1_.get_dimension());

  // compute slip_np1
  CP::updateSlip<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    state_internal_.resistance_,
    state_internal_.shear_np1_,
    state_internal_.slip_n_,
    slip_computed,
    failed);

  residual = state_internal_.slip_np1_ - slip_computed;

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "residual" << std::endl;
    std::cout << residual << std::endl;
  }

  this->norm_residual_ = Sacado::ScalarValue<ScalarT>::eval(norm(residual));

  if (dt_ > 0.0) {
    state_internal_.rates_slip_ =
      (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  return;
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::ImplicitIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitIntegrator(
      Minimizer const & minimizer,
      RolMinimizer const & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity)
  : Base(
      nox_status_test,
      slip_systems,
      slip_families,
      state_mechanical,
      state_internal,
      C,
      dt,
      verbosity),
    minimizer_(minimizer), rol_minimizer_(rol_minimizer), step_type_(step_type), using_rol_minimizer_(false)
{

}

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::ImplicitIntegrator<EvalT, NumDimT, NumSlipT>::reevaluateState() const
{
  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << ">>> in ImplicitIntegrator::reevaluateState" << std::endl;
  }

  std::ofstream outfile;
  std::stringstream ss;
  ss << "slips_" << state_internal_.cell_
     << "_" << state_internal_.pt_ <<  ".out";
  std::string file = ss.str();

  if (state_internal_.cell_ != -1) {
    outfile.open(file, std::fstream::app);
  }

  bool
  minimizer_failed(minimizer_.failed);

  bool
  minimizer_converged(minimizer_.converged);

  std::string
  minimizer_failure_message(minimizer_.failure_message);

  int
  minimizer_num_iter(minimizer_.num_iter); // not defined for rol minimizer

  if(using_rol_minimizer_){
    minimizer_failed = rol_minimizer_.failed;
    minimizer_converged = rol_minimizer_.converged;
    minimizer_failure_message = rol_minimizer_.failure_message;
  }

  // cases in which the model is subject to divergence
  // more work to do in the nonlinear systems
  if(minimizer_failed){
    // TODO: reenable printout
    /*if (verbosity_ > 2){
      std::cout << "\n**** CrystalPlasticityModel computeState() ";
      std::cout << "exited due to failure criteria.\n" << std::endl;
    }*/
    if (state_internal_.cell_ != -1) {
      outfile << "minimizer failed" << "\n";
      outfile.close();
    }
    this->forceGlobalLoadStepReduction(minimizer_failure_message);
    return;
  }

  if(!minimizer_converged){
    // TODO: renable printout
    /*if(verbosity_ > 2){
      std::cout << "\n**** CrystalPlasticityModel computeState()";
      std::cout << " failed to converge.\n" << std::endl;
      rol_minimizer_.printReport(std::cout);
    }*/
    if (state_internal_.cell_ != -1) {
      outfile << "minimizer not converged" << "\n";
      outfile.close();
    }
    this->forceGlobalLoadStepReduction("Minisolver not converged");
    return;
  }

  // We now have the solution for slip_np1, including sensitivities
  // (if any). Re-evaluate all the other state variables based on slip_np1.
  // Compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    dt_,
    state_internal_.slip_n_,
    state_internal_.slip_np1_,
    state_mechanical_.Fp_n_,
    state_mechanical_.Lp_np1_,
    state_mechanical_.Fp_np1_);

  bool
  failed{false};

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    C_,
    state_mechanical_.F_np1_,
    state_mechanical_.Fp_np1_,
    state_mechanical_.sigma_np1_,
    state_mechanical_.S_np1_,
    state_internal_.shear_np1_,
    failed);

  if(failed == true){
    this->forceGlobalLoadStepReduction("ComputeStress failed.");
    return;
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_shear{0.0};

    max_shear = Sacado::ScalarValue<ScalarT>::eval(
	      minitensor::norm_infinity(state_internal_.shear_np1_));

    outfile << max_shear << "\n";
    outfile.close();
  }

  if (dt_ > 0.0) {
    state_internal_.rates_slip_ =
      (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Compute the residual norm
  if(!using_rol_minimizer_){
    this->norm_residual_ = std::sqrt(2.0 * minimizer_.final_value);
  }
  else{
    this->norm_residual_ = std::sqrt(2.0 * rol_minimizer_.final_value);
  }

  this->num_iters_ = minimizer_num_iter;

  return;
}

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitSlipIntegrator(
      Minimizer const & minimizer,
      RolMinimizer const & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity)
  : Base(
      minimizer,
      rol_minimizer,
      step_type,
      nox_status_test,
      slip_systems,
      slip_families,
      state_mechanical,
      state_internal,
      C,
      dt,
      verbosity)
{
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>::update() const
{
  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << ">>> in ImplicitSlipIntegrator::update" << std::endl;
  }

  // Unknown for solver
  minitensor::Vector<ScalarT, CP::NlsDim<NumSlipT>::value>
  x(this->num_slip_);

  using NonlinearSolver = CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>;

  NonlinearSolver
  nls(
      C_,
      slip_systems_,
      slip_families_,
      state_mechanical_.Fp_n_,
      state_internal_.hardening_n_,
      state_internal_.slip_n_,
      state_mechanical_.F_np1_,
      dt_,
      verbosity_);

  CP::Dissipation<NumDimT, NumSlipT, EvalT>
  initial_guess_nls(
      slip_systems_,
      slip_families_,
      state_internal_.hardening_n_,
      state_internal_.slip_n_,
      state_mechanical_.F_n_,
      state_mechanical_.F_np1_,
      dt_,
      verbosity_);

  using StepType = minitensor::StepBase<NonlinearSolver, ValueT, CP::NlsDim<NumSlipT>::value>;

  std::unique_ptr<StepType>
  pstep = minitensor::stepFactory<NonlinearSolver, ValueT, CP::NlsDim<NumSlipT>::value>(step_type_);

  // Initial guess for x should be slip_np1
  for (int i = 0; i < this->num_slip_; ++i) {
    x(i) = Sacado::ScalarValue<ScalarT>::eval(state_internal_.slip_np1_(i));
  }

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "x_{n+1}^(0)" << std::endl;
    std::cout << x << std::endl;
    std::cout << "F_{n+1}" << std::endl;
    std::cout << state_mechanical_.F_np1_ << std::endl;
    std::cout << "Fp_{n}" << std::endl;
    std::cout << state_mechanical_.Fp_n_ << std::endl;
  }

  LCM::MiniSolver<Minimizer, StepType, NonlinearSolver, EvalT, CP::NlsDim<NumSlipT>::value>
  mini_solver(minimizer_, *pstep, nls, x);

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "x_{n+1}" << std::endl;
    std::cout << x << std::endl;
  }

  if (minimizer_.failed == true)
  {
    this->forceGlobalLoadStepReduction(minimizer_.failure_message);
    return;
  }

  // Write slip back out from x
  for (int i = 0; i < this->num_slip_; ++i) {
    state_internal_.slip_np1_[i] = x[i];
    state_internal_.hardening_np1_[i] = state_internal_.hardening_n_[i];
  }

  minitensor::Vector<ScalarT, NumSlipT>
  slip_rate(state_internal_.rates_slip_.get_dimension());

  slip_rate.fill(minitensor::Filler::ZEROS);
  if (dt_ > 0.0) {
    slip_rate = (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Update hardness
  bool hardness_failed = false;
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_rate,
      state_internal_.hardening_n_,
      state_internal_.hardening_np1_,
      state_internal_.resistance_,
      hardness_failed);

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "resistance_{n+1}" << std::endl;
    std::cout << state_internal_.resistance_ << std::endl;
  }

  if (hardness_failed) {
    this->forceGlobalLoadStepReduction("Failed on hardness");
    return;
  }

  this->reevaluateState();
  return;
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::ImplicitSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitSlipHardnessIntegrator(
      const Minimizer & minimizer,
      const RolMinimizer & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity)
  : Base(
      minimizer,
      rol_minimizer,
      step_type,
      nox_status_test,
      slip_systems,
      slip_families,
      state_mechanical,
      state_internal,
      C,
      dt,
      verbosity)
{
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::ImplicitSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>::update() const
{
  // Unknown for solver
  minitensor::Vector<ScalarT, CP::NlsDim<NumSlipT>::value>
  x(this->num_slip_ * 2);

  using NonlinearSystem = CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>;

  NonlinearSystem
  nls(
      C_,
      slip_systems_,
      slip_families_,
      state_mechanical_.Fp_n_,
      state_internal_.hardening_n_,
      state_internal_.slip_n_,
      state_mechanical_.F_np1_,
      dt_,
      verbosity_);

  using StepType = minitensor::StepBase<NonlinearSystem, ValueT, CP::NlsDim<NumSlipT>::value>;

  std::unique_ptr<StepType>
  pstep = minitensor::stepFactory<NonlinearSystem, ValueT, CP::NlsDim<NumSlipT>::value>(step_type_);

  std::ofstream outfile;
  std::stringstream ss;
  ss << "slips_" << state_internal_.cell_
     << "_" << state_internal_.pt_ <<  ".out";
  std::string file = ss.str();

  if (state_internal_.cell_ != -1) {
    outfile.open(file, std::fstream::app);
  }

  for (int i = 0; i < this->num_slip_; ++i) {
    // initial guess for x(0:this->num_slip_-1) from predictor
    x(i) = Sacado::ScalarValue<ScalarT>::eval(state_internal_.slip_np1_(i));
    // initial guess for x(this->num_slip_:2*this->num_slip_) from predictor
    x(i + this->num_slip_) =
    Sacado::ScalarValue<ScalarT>::eval(state_internal_.hardening_np1_(i));
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_slip{0.0};

    RealType
    tot_slip{0.0};

    max_slip = Sacado::ScalarValue<ScalarT>::eval(
	     minitensor::norm_infinity(state_internal_.slip_np1_));

    tot_slip = Sacado::ScalarValue<ScalarT>::eval(
	     minitensor::norm_1(state_internal_.slip_np1_));

    outfile  << dt_ << ", " << max_slip << ", " << tot_slip << ", ";
  }

  LCM::MiniSolver<Minimizer, StepType, NonlinearSystem, EvalT, CP::NlsDim<NumSlipT>::value>
  mini_solver(minimizer_, *pstep, nls, x);

  if (minimizer_.warning == true) {
    // std::cout << "MiniSolver WARNING: " << minimizer_.warning_message << std::endl;
    this->setWarningStatus(minimizer_.warning_message);
  }

  if (minimizer_.failed == true)
  {
    if (state_internal_.cell_ != -1) {
      outfile << "minimizer failed" << "\n";
      outfile.close();
    }
    this->forceGlobalLoadStepReduction(minimizer_.failure_message);
    return;
  }

  for(int i=0; i<this->num_slip_; ++i) {
    state_internal_.slip_np1_[i] = x[i];
    state_internal_.hardening_np1_[i] = x[i + this->num_slip_];
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_slip{0.0};

    RealType
    tot_slip{0.0};

    max_slip = Sacado::ScalarValue<ScalarT>::eval(
             minitensor::norm_infinity(state_internal_.slip_np1_));

    tot_slip = Sacado::ScalarValue<ScalarT>::eval(
             minitensor::norm_1(state_internal_.slip_np1_));

    outfile << max_slip << ", " << tot_slip << ", ";
  }

  minitensor::Vector<ScalarT, NumSlipT>
  slip_rate(this->num_slip_);

  slip_rate.fill(minitensor::Filler::ZEROS);
  if (dt_ > 0.0) {
    slip_rate = (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Update hardness
  bool hardness_failed = false;
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_rate,
      state_internal_.hardening_n_,
      state_internal_.hardening_np1_,
      state_internal_.resistance_,
      hardness_failed);

  if (hardness_failed) {
    this->forceGlobalLoadStepReduction("Failed on hardness");
    return;
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_hard{0.0};

    max_hard = Sacado::ScalarValue<ScalarT>::eval(
	     minitensor::norm_infinity(state_internal_.hardening_np1_));

    outfile << max_hard << ", ";
    outfile.close();
  }

  this->reevaluateState();
  return;
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::ImplicitConstrainedSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitConstrainedSlipHardnessIntegrator(
      const Minimizer & minimizer,
      const RolMinimizer & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity)
  : Base(
      minimizer,
      rol_minimizer,
      step_type,
      nox_status_test,
      slip_systems,
      slip_families,
      state_mechanical,
      state_internal,
      C,
      dt,
      verbosity)
{
  this->using_rol_minimizer_ = true;
}


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::ImplicitConstrainedSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>::update() const
{
  // Unknown for solver
  minitensor::Vector<ScalarT, CP::NlsDim<NumSlipT>::value>
  x(this->num_slip_ * 2);

  using FN = CP::ResidualSlipHardnessFN<NumDimT, NumSlipT, EvalT>;

  FN
  fn(
      C_,
      slip_systems_,
      slip_families_,
      state_mechanical_.Fp_n_,
      state_internal_.hardening_n_,
      state_internal_.slip_n_,
      state_mechanical_.F_np1_,
      dt_,
      verbosity_);

  std::ofstream outfile;
  std::stringstream ss;
  ss << "slips_" << state_internal_.cell_
     << "_" << state_internal_.pt_ <<  ".out";
  std::string file = ss.str();

  if (state_internal_.cell_ != -1) {
    outfile.open(file, std::fstream::app);
  }

  for (int i = 0; i < this->num_slip_; ++i) {
    // initial guess for x(0:this->num_slip_-1) from predictor
    x(i) = Sacado::ScalarValue<ScalarT>::eval(state_internal_.slip_np1_(i));
    // initial guess for x(this->num_slip_:2*this->num_slip_) from predictor
    x(i + this->num_slip_) =
    Sacado::ScalarValue<ScalarT>::eval(state_internal_.hardening_n_(i));
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_slip{0.0};

    RealType
    tot_slip{0.0};

    max_slip = Sacado::ScalarValue<ScalarT>::eval(
	     minitensor::norm_infinity(state_internal_.slip_np1_));

    tot_slip = Sacado::ScalarValue<ScalarT>::eval(
	     minitensor::norm_1(state_internal_.slip_np1_));

    outfile  << dt_ << ", " << max_slip << ", " << tot_slip << ", ";
  }

  using BC = minitensor::Bounds<ValueT, CP::NlsDim<NumSlipT>::value>;

  minitensor::Vector<ValueT, CP::NlsDim<NumSlipT>::value> lo, hi;
  lo.set_dimension(this->num_slip_ * 2);
  hi.set_dimension(this->num_slip_ * 2);
  for(int i=0; i<this->num_slip_; ++i) {
    lo[i] = -1.0e50;
    hi[i] = 1.0e50;
    lo[i + this->num_slip_] = -1.0e-50;
    hi[i + this->num_slip_] = 1.0e-50;
  }
  BC bounds(lo, hi);

  // Define algorithm.
  std::string const
  algoname{"Line Search"};

  // Set parameters.
  Teuchos::ParameterList
  params;

  params.sublist("Step").sublist("Line Search").sublist("Descent Method").
    set("Type", "Newton-Krylov");

  params.sublist("Status Test").set("Gradient Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Step Tolerance", 1.0e-16);
  params.sublist("Status Test").set("Iteration Limit", 128);

  LCM::MiniSolverBoundsROL<RolMinimizer, FN, BC, EvalT, CP::NlsDim<NumSlipT>::value>
  mini_solver(rol_minimizer_, algoname, params, fn, bounds, x);

  //  rol_minimizer_.printReport(std::cout);

  if (rol_minimizer_.failed == true)
  {
    if (state_internal_.cell_ != -1) {
      outfile << "rol_minimizer_ failed" << "\n";
      outfile.close();
    }
    this->forceGlobalLoadStepReduction(rol_minimizer_.failure_message);
    return;
  }

  for(int i=0; i<this->num_slip_; ++i) {
    state_internal_.slip_np1_[i] = x[i];
    state_internal_.hardening_np1_[i] = x[i + this->num_slip_];
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_slip{0.0};

    RealType
    tot_slip{0.0};

    max_slip = Sacado::ScalarValue<ScalarT>::eval(
             minitensor::norm_infinity(state_internal_.slip_np1_));

    tot_slip = Sacado::ScalarValue<ScalarT>::eval(
             minitensor::norm_1(state_internal_.slip_np1_));

    outfile << max_slip << ", " << tot_slip << ", ";
  }

  minitensor::Vector<ScalarT, NumSlipT>
  slip_rate(this->num_slip_);

  slip_rate.fill(minitensor::Filler::ZEROS);
  if (dt_ > 0.0) {
    slip_rate = (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Update hardness
  bool hardness_failed = false;
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_rate,
      state_internal_.hardening_n_,
      state_internal_.hardening_np1_,
      state_internal_.resistance_,
      hardness_failed);

  if (hardness_failed) {
    this->forceGlobalLoadStepReduction("Failed on hardness");
    return;
  }

  if (state_internal_.cell_ != -1) {
    RealType
    max_hard{0.0};

    max_hard = Sacado::ScalarValue<ScalarT>::eval(
	     minitensor::norm_infinity(state_internal_.hardening_np1_));

    outfile << max_hard << ", ";
    outfile.close();
  }

  this->reevaluateState();
  return;
}
