//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::Integrator<EvalT, NumDimT, NumSlipT>::forceGlobalLoadStepReduction() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      nox_status_test_.is_null(),
      std::logic_error,
      "\n**** Error in CrystalPlasticityModel: \
          error accessing NOX status test.");

  nox_status_test_->status_ = NOX::StatusTest::Failed;
}

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::IntegratorFactory<EvalT, NumDimT, NumSlipT>::IntegratorFactory(
      utility::StaticAllocator & allocator,
      Minimizer const & minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      CP::StateMechanical<ScalarT, NumDimT> & state_mechanical,
      CP::StateInternal<ScalarT, NumSlipT> & state_internal,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : allocator_(allocator),
    minimizer_(minimizer),
    step_type_(step_type),
    nox_status_test_(nox_status_test),
    slip_systems_(slip_systems),
    slip_families_(slip_families),
    state_mechanical_(state_mechanical),
    state_internal_(state_internal),
    C_(C),
    F_np1_(F_np1),
    dt_(dt)
{}

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
utility::StaticPointer<typename CP::IntegratorFactory<EvalT, NumDimT, NumSlipT>::IntegratorBase>
CP::IntegratorFactory<EvalT, NumDimT, NumSlipT>::operator()(
    CP::IntegrationScheme integration_scheme,
    CP::ResidualType residual_type) const
{
  switch (integration_scheme)
  {
    case CP::IntegrationScheme::EXPLICIT:
    {
      using IntegratorType = CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>;
      return allocator_.create<IntegratorType>(nox_status_test_,
          slip_systems_,
          slip_families_,
          state_mechanical_,
          state_internal_,
          C_,
          F_np1_,
          dt_);

    } break;

    case CP::IntegrationScheme::IMPLICIT:
    {
      switch (residual_type)
      {
        case CP::ResidualType::SLIP:
        {
          using IntegratorType
            = CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>;
          return allocator_.create<IntegratorType>(minimizer_,
              step_type_,
              nox_status_test_,
              slip_systems_,
              slip_families_,
              state_mechanical_,
              state_internal_,
              C_,
              F_np1_,
              dt_);
        } break;

        case CP::ResidualType::SLIP_HARDNESS:
        {
          using IntegratorType
            = CP::ImplicitSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>;
          return allocator_.create<IntegratorType>(minimizer_,
              step_type_,
              nox_status_test_,
              slip_systems_,
              slip_families_,
              state_mechanical_,
              state_internal_,
              C_,
              F_np1_,
              dt_);
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
    
template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>::ExplicitIntegrator(
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : Base(nox_status_test, slip_systems, slip_families, state_mechanical, state_internal, C, F_np1, dt)
{

}

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
bool
CP::ExplicitIntegrator<EvalT, NumDimT, NumSlipT>::update(
    RealType & residual_norm) const
{
  Intrepid2::Vector<ScalarT, NumSlipT>
  residual(this->num_slip_);

  Intrepid2::Tensor<ScalarT, NumDimT>
  Fp_n_FAD(state_mechanical_.Fp_n_.get_dimension());

  for (auto i = 0; i < state_mechanical_.Fp_n_.get_number_components(); ++i) {
    Fp_n_FAD[i] = state_mechanical_.Fp_n_[i];
  }

  // compute sigma_np1, S_np1, and shear_np1 using Fp_n
  CP::computeStress<NumDimT, NumSlipT, ScalarT>(
    slip_systems_, 
    C_, 
    F_np1_, 
    Fp_n_FAD, 
    state_mechanical_.sigma_np1_, 
    state_mechanical_.S_np1_, 
    state_internal_.shear_np1_);

  // compute state_hardening_np1 using slip_n
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    state_internal_.rate_slip_,
    state_internal_.hardening_n_, 
    state_internal_.hardening_np1_,
    state_internal_.resistance_);

  // compute slip_np1
  CP::updateSlip<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    state_internal_.resistance_,
    state_internal_.shear_np1_,
    state_internal_.slip_n_,
    state_internal_.slip_np1_);

  // compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT, ScalarT>(
    slip_systems_, 
    dt_,
    state_internal_.slip_n_,
    state_internal_.slip_np1_,
    state_mechanical_.Fp_n_,
    state_mechanical_.Lp_np1_, 
    state_mechanical_.Fp_np1_);

  // compute sigma_np1, S_np1, and shear_np1 using Fp_np1
  CP::computeStress<NumDimT, NumSlipT, ScalarT>(
    slip_systems_, 
    C_, 
    F_np1_, 
    state_mechanical_.Fp_np1_, 
    state_mechanical_.sigma_np1_, 
    state_mechanical_.S_np1_, 
    state_internal_.shear_np1_);

  Intrepid2::Vector<ScalarT, NumSlipT>
  slip_computed(state_internal_.slip_np1_.get_dimension());

  // compute slip_np1
  CP::updateSlip<NumDimT, NumSlipT, ScalarT>(
    slip_systems_,
    slip_families_,
    dt_,
    state_internal_.resistance_,
    state_internal_.shear_np1_,
    state_internal_.slip_n_,
    slip_computed);

  residual = state_internal_.slip_np1_ - slip_computed;

  residual_norm = Sacado::ScalarValue<ScalarT>::eval(norm(residual));

  if (dt_ > 0.0) {
    state_internal_.rate_slip_ = 
      (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  return true;
}


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::ImplicitIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitIntegrator(
      const Minimizer & minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : Base(nox_status_test, slip_systems, slip_families, state_mechanical, state_internal, C, F_np1, dt),
  minimizer_(minimizer), step_type_(step_type)
{

}

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
bool
CP::ImplicitIntegrator<EvalT, NumDimT, NumSlipT>::reevaluateState(RealType & residual_norm) const
{
  if(!minimizer_.converged){
    // TODO: renable printout
    /*if(verbosity_ > 2){
      std::cout << "\n**** CrystalPlasticityModel computeState()";
      std::cout << " failed to converge.\n" << std::endl;
      minimizer_.printReport(std::cout);
    }*/
    this->forceGlobalLoadStepReduction();
    return false;
  }

  // cases in which the model is subject to divergence
  // more work to do in the nonlinear systems
  if(minimizer_.failed){
    // TODO: reenable printout
    /*if (verbosity_ > 2){
      std::cout << "\n**** CrystalPlasticityModel computeState() ";
      std::cout << "exited due to failure criteria.\n" << std::endl;
    }*/
    this->forceGlobalLoadStepReduction();
    return false;
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

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT, ScalarT>(
    slip_systems_, 
    C_, 
    F_np1_, 
    state_mechanical_.Fp_np1_, 
    state_mechanical_.sigma_np1_, 
    state_mechanical_.S_np1_, 
    state_internal_.shear_np1_);

  if (dt_ > 0.0) {
    state_internal_.rate_slip_ = 
      (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Compute the residual norm 
  residual_norm = std::sqrt(2.0 * minimizer_.final_value);
  
  this->num_iters_ = minimizer_.num_iter;

  return true;
}

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitSlipIntegrator(
      const Minimizer & minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : Base(minimizer, step_type, nox_status_test, slip_systems,
         slip_families, state_mechanical, state_internal, C, F_np1, dt)
{
}


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
bool
CP::ImplicitSlipIntegrator<EvalT, NumDimT, NumSlipT>::update(
    RealType & residual_norm) const
{
  // Unknown for solver
  Intrepid2::Vector<ScalarT, CP::NlsDim<NumSlipT>::value>
  x(this->num_slip_);

  using NonlinearSolver = CP::ResidualSlipNLS<NumDimT, NumSlipT, EvalT>; 

  NonlinearSolver nls(C_, slip_systems_, slip_families_, state_mechanical_.Fp_n_,
                      state_internal_.hardening_n_, state_internal_.slip_n_,
                      F_np1_, dt_);

  using StepType = Intrepid2::StepBase<NonlinearSolver, ValueT, CP::NlsDim<NumSlipT>::value>;

  std::unique_ptr<StepType>
  pstep = Intrepid2::stepFactory<NonlinearSolver, ValueT, CP::NlsDim<NumSlipT>::value>(step_type_);
  
  // Initial guess for x should be slip_np1
  for (int i = 0; i < this->num_slip_; ++i) {
    x(i) = Sacado::ScalarValue<ScalarT>::eval(state_internal_.slip_np1_(i));
  }

  LCM::MiniSolver<Minimizer, StepType, NonlinearSolver, EvalT, CP::NlsDim<NumSlipT>::value>
  mini_solver(minimizer_, *pstep, nls, x);

  // Write slip back out from x
  for (int i = 0; i < this->num_slip_; ++i) {
    state_internal_.slip_np1_(i) = x(i);
  }

  Intrepid2::Vector<ScalarT, NumSlipT>
  slip_rate(state_internal_.rate_slip_.get_dimension());

  slip_rate.fill(Intrepid2::ZEROS);
  if (dt_ > 0.0) {
    slip_rate = (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Update hardness
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_rate,
      state_internal_.hardening_n_,
      state_internal_.hardening_np1_,
      state_internal_.resistance_);

  return this->reevaluateState(residual_norm);
}


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
CP::ImplicitSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>::ImplicitSlipHardnessIntegrator(
      const Minimizer & minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
  : Base(minimizer, step_type, nox_status_test, slip_systems,
         slip_families, state_mechanical, state_internal, C, F_np1, dt)
{
}


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
bool
CP::ImplicitSlipHardnessIntegrator<EvalT, NumDimT, NumSlipT>::update(
    RealType & residual_norm) const
{
  // Unknown for solver
  Intrepid2::Vector<ScalarT, CP::NlsDim<NumSlipT>::value>
  x(this->num_slip_ * 2);

  using NonlinearSolver = CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>; 

  NonlinearSolver nls(C_, slip_systems_, slip_families_, state_mechanical_.Fp_n_,
                      state_internal_.hardening_n_, state_internal_.slip_n_,
                      F_np1_, dt_);

  using StepType = Intrepid2::StepBase<NonlinearSolver, ValueT, CP::NlsDim<NumSlipT>::value>;

  std::unique_ptr<StepType>
  pstep = Intrepid2::stepFactory<NonlinearSolver, ValueT, CP::NlsDim<NumSlipT>::value>(step_type_);

  for (int i = 0; i < this->num_slip_; ++i) {
    // initial guess for x(0:this->num_slip_-1) from predictor
    x(i) = Sacado::ScalarValue<ScalarT>::eval(state_internal_.slip_np1_(i));
    // initial guess for x(this->num_slip_:2*this->num_slip_) from predictor
    x(i + this->num_slip_) = 
    Sacado::ScalarValue<ScalarT>::eval(state_internal_.hardening_n_(i));
  }
  
  LCM::MiniSolver<Minimizer, StepType, NonlinearSolver, EvalT, CP::NlsDim<NumSlipT>::value>
  mini_solver(minimizer_, *pstep, nls, x);

  for(int i=0; i<this->num_slip_; ++i) {
    state_internal_.slip_np1_[i] = x[i];
    state_internal_.hardening_np1_[i] = x[i + this->num_slip_];
  }

  Intrepid2::Vector<ScalarT, NumSlipT>
  slip_rate(this->num_slip_);

  slip_rate.fill(Intrepid2::ZEROS);
  if (dt_ > 0.0) {
    slip_rate = (state_internal_.slip_np1_ - state_internal_.slip_n_) / dt_;
  }

  // Update hardness
  CP::updateHardness<NumDimT, NumSlipT, ScalarT>(
      slip_systems_,
      slip_families_,
      dt_,
      slip_rate,
      state_internal_.hardening_n_,
      state_internal_.hardening_np1_,
      state_internal_.resistance_);

  return this->reevaluateState(residual_norm);
}

