//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>

///
/// Verify that constitutive update has preserved finite values
///
template<Intrepid2::Index NumDimT, typename ArgT>
void
CP::confirmTensorSanity(
    Intrepid2::Tensor<ArgT, NumDimT> const & input,
    std::string const & message)
{
  int dim = input.get_dimension();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      assert(boost::math::isfinite(Sacado::ScalarValue<ArgT>::eval(input(i, j)))==true);
      // Disabling this capability for release.
      // We will revisit this option when we can cut the time step from the constitutive model.
      /* if (!boost::math::isfinite(
          Sacado::ScalarValue<ArgT>::eval(input(i, j)))) {
        std::string msg =
            "**** Invalid data detected in CP::confirmTensorSanity(): "
                + message;
        TEUCHOS_TEST_FOR_EXCEPTION(
            !boost::math::isfinite(
                Sacado::ScalarValue<ArgT>::eval(input(i, j))),
            std::logic_error,
            msg);
      } */
    }
  }
}



///
/// Update the plastic quantities
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
CP::applySlipIncrement(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
    Intrepid2::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid2::Tensor<ArgT, NumDimT> & Fp_np1)
{
  Intrepid2::Index const
  num_slip = slip_n.get_dimension();

  Intrepid2::Index const
  num_dim = Fp_n.get_dimension();

  // 
  // calculate plastic velocity gradient
  //
  Intrepid2::Tensor<ArgT, NumDimT>
  exp_L_dt(num_dim);

  Lp_np1.fill(Intrepid2::ZEROS);

  if(dt > 0){
    for (int s(0); s < num_slip; ++s) {
      Lp_np1 += (slip_np1[s] - slip_n[s])/dt * slip_systems[s].projector_;
    }
  }

  CP::confirmTensorSanity<NumDimT>(Lp_np1, "Lp_np1 in applySlipIncrement().");

  // update plastic deformation gradient
  // F^{p}_{n+1} = exp(L_{n+1} * delta t) F^{p}_{n}
  exp_L_dt = Intrepid2::exp(Lp_np1 * dt);
  Fp_np1 = exp_L_dt * Fp_n;

  CP::confirmTensorSanity<NumDimT>(Fp_np1, "Fp_np1 in applySlipIncrement()");
}




///
/// Evolve the hardnesses
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
CP::updateHardness(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  for (int sf_index(0); sf_index < slip_families.size(); ++ sf_index)
  {
    auto const &
    slip_family = slip_families[sf_index];

    CP::hardeningLawFactory<NumDimT, NumSlipT, ArgT>
    f;

    CP::HardeningLawType const
    type_hardening_law = slip_family.type_hardening_law_;

    CP::HardeningLawBase<NumDimT, NumSlipT, ArgT> *
    phardening = f.createHardeningLaw(type_hardening_law);

    phardening->harden(
      slip_family,
      slip_systems,
      dt, 
      rate_slip, 
      state_hardening_n, 
      state_hardening_np1, 
      slip_resistance);
  }

  return;
}




///
/// Update the plastic slips
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
CP::updateSlip(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_resistance,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1)
{
  for (int ss_index(0); ss_index < slip_systems.size(); ++ ss_index)
  {
    auto const &
    slip_family = slip_families[slip_systems[ss_index].slip_family_index_];

    CP::flowRuleFactory<NumDimT, NumSlipT, ArgT>
    f;

    CP::FlowRuleBase<NumDimT, NumSlipT, ArgT> *
    pflow = f.createFlowRule(slip_family.type_flow_rule_);

    ArgT const
    rate_slip = pflow->computeRateSlip(
        slip_family.pflow_parameters_,
        shear[ss_index],
        slip_resistance[ss_index]);

    slip_np1[ss_index] = slip_n[ss_index] + dt * rate_slip;
  }

  return;
}



///
/// Compute the stresses 
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT, typename DataT>
void
CP::computeStress(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    Intrepid2::Tensor4<DataT, NumDimT> const & C,
    Intrepid2::Tensor<DataT, NumDimT> const & F,
    Intrepid2::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear)
{
  Intrepid2::Index const
  num_dim = F.get_dimension();

  Intrepid2::Index const
  num_slip = shear.get_dimension();

  Intrepid2::Tensor<ArgT, NumDimT>
  defgrad_elastic(num_dim);

  Intrepid2::Tensor<ArgT, NumDimT>
  strain_elastic(num_dim);

  Intrepid2::Tensor<ArgT, NumDimT>
  deformation_elastic(num_dim);

  // Saint Venant–Kirchhoff model
  if (Intrepid2::det(Fp) == 0.0)
  {
    std::cout << "Singular plastic deformation gradient" << std::endl;
    std::cout << std::setprecision(4) << Fp << std::endl;
  }

  defgrad_elastic = F * Intrepid2::inverse(Fp);

  deformation_elastic = Intrepid2::transpose(defgrad_elastic) * defgrad_elastic;

  strain_elastic = 
    0.5 * (deformation_elastic - Intrepid2::identity<ArgT, NumDimT>(num_dim));

  S = Intrepid2::dotdot(C, strain_elastic);

  sigma = 1.0 / Intrepid2::det(defgrad_elastic) * 
    defgrad_elastic * S * Intrepid2::transpose(defgrad_elastic);

  CP::confirmTensorSanity<NumDimT>(
      sigma,
      "Cauchy stress in ResidualSlipNLS::computeStress()");

  // Compute resolved shear stresses
  for (int s(0); s < num_slip; ++s) {
    shear[s] = 
      Intrepid2::dotdot(slip_systems[s].projector_, deformation_elastic * S);
  }
}



//
//! Construct elasticity tensor
//
template<Intrepid2::Index NumDimT, typename DataT, typename ArgT>
void
CP::computeCubicElasticityTensor(
    DataT c11, 
    DataT c12, 
    DataT c44,
    Intrepid2::Tensor4<ArgT, NumDimT> & C)
{

  Intrepid2::Index const
  num_dim = C.get_dimension();

  C.fill(Intrepid2::ZEROS);

  for (Intrepid2::Index dim_i = 0; dim_i < num_dim; ++dim_i) {
    C(dim_i, dim_i, dim_i, dim_i) = c11;
    for (Intrepid2::Index dim_j = dim_i + 1; dim_j < num_dim; ++dim_j) {
      C(dim_i, dim_i, dim_j, dim_j) = c12;
      C(dim_j, dim_j, dim_i, dim_i) = C(dim_i, dim_i, dim_j, dim_j);
      C(dim_i, dim_j, dim_i, dim_j) = c44;
      C(dim_j, dim_i, dim_j, dim_i) = C(dim_i, dim_j, dim_i, dim_j);
      C(dim_i, dim_j, dim_j, dim_i) = C(dim_i, dim_j, dim_i, dim_j);
      C(dim_j, dim_i, dim_i, dim_j) = C(dim_i, dim_j, dim_i, dim_j);
    }
  }
}











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
// Flow rules
//



//
// Power law flow rule
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
ScalarT
CP::PowerLawFlowRule<NumDimT, NumSlipT, ScalarT>::
computeRateSlip(
  std::unique_ptr<CP::FlowParameterBase> pflow_parameters,
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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
ScalarT
CP::ThermalActivationFlowRule<NumDimT, NumSlipT, ScalarT>::
computeRateSlip(
  std::unique_ptr<CP::FlowParameterBase> pflow_parameters,
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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
ScalarT
CP::PowerLawDragFlowRule<NumDimT, NumSlipT, ScalarT>::
computeRateSlip(
  std::unique_ptr<CP::FlowParameterBase> pflow_parameters,
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
  viscous_drag = std::fabs(ratio_stress) / drag_term;

  RealType const
  pl_tol = std::pow(2.0 * std::numeric_limits<RealType>::min(), 0.5 / m);

  bool const
  finite_power_law = std::fabs(ratio_stress) > pl_tol;

  ScalarT
  power_law{0.0};

  if (finite_power_law == true) {
    power_law = std::pow(std::fabs(ratio_stress), m - 1) * ratio_stress;
  }

  RealType const
  eff_tol = 1.0e-8;

  bool const
  vd_active = std::fabs(ratio_stress) > eff_tol;

  // prevent flow rule singularities if stress is zero
  ScalarT
  effective{power_law};

  if (vd_active == true) {
    effective = 1.0/((1.0 / power_law) + (1.0 / viscous_drag));
  }

  // compute slip increment
  return  g0 * effective;

}

//
// No flow rule
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
ScalarT
CP::NoFlowRule<NumDimT, NumSlipT, ScalarT>::
computeRateSlip(
  std::unique_ptr<CP::FlowParameterBase> pflow_parameters,
  ScalarT const & shear,
  ScalarT const & slip_resistance)
{
  return 0.;
}



//
// Hardening Laws
//

//
// Linear hardening with recovery
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::LinearMinusRecoveryHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(Intrepid2::ONES);

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
void
CP::LinearMinusRecoveryHardeningLaw<NumDimT, NumSlipT, ScalarT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance)
{
  using Params = LinearMinusRecoveryHardeningParameters<NumDimT, NumSlipT>;

  auto const
  num_slip_sys{slip_family.num_slip_sys_};

  Intrepid2::Vector<ScalarT, NumSlipT>
  rate_slip_abs(num_slip_sys);

  for (Intrepid2::Index ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_family.slip_system_indices_(ss_index);

    auto const &
    slip_rate = rate_slip(ss_index_global);

    rate_slip_abs(ss_index) = std::fabs(slip_rate);
  }

  Intrepid2::Vector<ScalarT, NumSlipT> const 
  driver_hardening = slip_family.latent_matrix_ * rate_slip_abs;

  auto const
  hardening_params = slip_family.phardening_parameters_;

  auto const
  param_map = hardening_params->param_map_;

  auto const
  modulus_recovery = hardening_params.getParameter(Params::MODULUS_RECOVERY);

  auto const
  modulus_hardening = hardening_params[param_map["Hardening Modulus"]];

  auto const
  resistance_slip_initial = hardening_params[param_map["Initial Slip Resistance"]];

  if (modulus_recovery > 0.0)
  {
    for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
    {
      auto const
      ss_index_global = slip_family.slip_system_indices_(ss_index);

      RealType const 
      effective_slip_n = -1.0 / modulus_recovery * 
        std::log(1.0 - modulus_recovery / modulus_hardening * state_hardening_n[ss_index_global]);

      state_hardening_np1[ss_index_global] = modulus_hardening / modulus_recovery * (1.0 - 
        std::exp(-modulus_recovery * (effective_slip_n + dt * driver_hardening[ss_index])));  

      slip_resistance[ss_index_global] = resistance_slip_initial + state_hardening_np1[ss_index_global];
    }
  }
  else
  {
    for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
    {
      auto const
      ss_index_global = slip_family.slip_system_indices_(ss_index);

      state_hardening_np1[ss_index_global] = 
        state_hardening_n[ss_index_global] + modulus_hardening * dt * driver_hardening[ss_index];

      slip_resistance[ss_index_global] = resistance_slip_initial + state_hardening_np1[ss_index_global];
    }
  }

//    state_hardening_np1[slip_sys] = state_hardening_n[slip_sys] +
//      dt * (H - Rd * (state_hardening_n[slip_sys])) * driver_hardening[slip_sys];

  return;
}

//
// Saturation hardening
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::SaturationHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(Intrepid2::ZEROS);

  for (int ss_index_i(0); ss_index_i < slip_family.num_slip_sys_; ++ss_index_i) {

    auto const
    slip_system_i = slip_systems[slip_family.slip_system_indices_[ss_index_i]];

    for (int ss_index_j(0); ss_index_j < slip_family.num_slip_sys_; ++ss_index_j) {

      auto const
      slip_system_j = slip_systems[slip_family.slip_system_indices_[ss_index_j]];

      slip_family.latent_matrix_(ss_index_i, ss_index_j) = 
        std::fabs(Intrepid2::dotdot(
          Intrepid2::sym(slip_system_i.projector_),
          Intrepid2::sym(slip_system_j.projector_)));
    }
  }

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
void
CP::SaturationHardeningLaw<NumDimT, NumSlipT, ScalarT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance)
{
  auto const
  num_slip_sys{slip_family.num_slip_sys_};

  Intrepid2::Vector<ScalarT, NumSlipT>
  rate_slip_abs(num_slip_sys);

  for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_family.slip_system_indices_(ss_index);

    auto const &
    slip_rate = rate_slip(ss_index_global);

    rate_slip_abs(ss_index) = std::fabs(slip_rate);
  }

  Intrepid2::Vector<ScalarT, NumSlipT> const 
  driver_hardening = slip_family.latent_matrix_ * rate_slip_abs;

  ScalarT
  effective_slip_rate{Intrepid2::norm_1(rate_slip)};

  auto const
  hardening_params = slip_family.phardening_parameters->hardening_params_;

  auto const
  param_map = slip_family.phardening_parameters->param_map_;

  auto const
  stress_saturation_initial = hardening_params[param_map["Initial Saturation Stress"]];

  auto const
  rate_slip_reference = hardening_params[param_map["Reference Slip Rate"]];

  auto const
  exponent_saturation = hardening_params[param_map["Saturation Exponent"]];

  auto const
  rate_hardening = hardening_params[param_map["Hardening Rate"]];

  auto const
  resistance_slip_initial = hardening_params[param_map["Initial Slip Resistancee"]];

  for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const &
    ss_index_global = slip_family.slip_system_indices_[ss_index];

    ScalarT
    stress_saturation{stress_saturation_initial};

    if (slip_family.exponent_saturation_ > 0.0) {
      stress_saturation = stress_saturation_initial * std::pow(
        effective_slip_rate / rate_slip_reference, exponent_saturation);
    }

    state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global] +
      dt * rate_hardening * driver_hardening[ss_index] *
      (stress_saturation - state_hardening_n[ss_index_global]) / 
      (stress_saturation - resistance_slip_initial);

    slip_resistance[ss_index_global] = state_hardening_np1[ss_index_global];
  } 

  return;
}

//
// Dislocation-density based hardening
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::DislocationDensityHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  Intrepid2::Index const
  num_dim = slip_systems[0].s_.get_dimension();

  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(Intrepid2::ZEROS);

  // std::cout << "latent_matrix" << std::endl;

  for (int ss_index_i(0); ss_index_i < slip_family.num_slip_sys_; ++ss_index_i)
  {
    auto const &
    slip_system_i = slip_systems[slip_family.slip_system_indices_[ss_index_i]];

    Intrepid2::Vector<RealType, CP::MAX_DIM>
    normal_i(num_dim);

    normal_i = slip_system_i.n_;

    for (int ss_index_j(0); ss_index_j < slip_family.num_slip_sys_; ++ss_index_j)
    {
      auto const &
      slip_system_j = slip_systems[slip_family.slip_system_indices_[ss_index_j]];

      Intrepid2::Vector<RealType, CP::MAX_DIM>
      direction_j = slip_system_j.s_;

      Intrepid2::Vector<RealType, CP::MAX_DIM>
      normal_j = slip_system_j.n_;

      Intrepid2::Vector<RealType, CP::MAX_DIM>
      transverse_j = Intrepid2::unit(Intrepid2::cross(normal_j, direction_j));

      slip_family.latent_matrix_(ss_index_i, ss_index_j) = 
          std::abs(Intrepid2::dot(normal_i, transverse_j));

      // std::cout << std::setprecision(3) << latent_matrix(ss_index_i, ss_index_j) << " ";
    }
    // std::cout << std::endl;
  }

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
void
CP::DislocationDensityHardeningLaw<NumDimT, NumSlipT, ScalarT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance)
{
  auto const
  num_slip_sys{slip_family.num_slip_sys_};

  //
  // Compute the effective dislocation density at step n
  //
  Intrepid2::Vector<RealType, NumSlipT>
  densities_forest = slip_family.latent_matrix_ * state_hardening_n;

  // std::cout << "densities_forest: " << densities_forest << std::endl;

  Intrepid2::Tensor<RealType, NumSlipT>
  aux_matrix(num_slip_sys);

  for (int ss_index_i(0); ss_index_i < num_slip_sys; ++ss_index_i)
  {
    for (int ss_index_j(0); ss_index_j < num_slip_sys; ++ss_index_j)
    {
      aux_matrix(ss_index_i, ss_index_j) =
          std::sqrt(1.0 - std::pow(slip_family.latent_matrix_(ss_index_i, ss_index_j), 2));
    }
  }

  Intrepid2::Vector<RealType, NumSlipT>
  state_hardening(num_slip_sys);

  for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_family.slip_system_indices_[ss_index];
  
    state_hardening[ss_index] = state_hardening_n[ss_index_global];
  }

  Intrepid2::Vector<RealType, NumSlipT>
  densities_parallel = aux_matrix * state_hardening;

  //
  // Update dislocation densities
  //
  auto const
  hardening_params = slip_family.phardening_parameters->hardening_params_;

  auto const
  param_map = slip_family.phardening_parameters->param_map_;

  RealType const
  factor_generation = hardening_params[param_map["Generation Factor"]];

  RealType const
  factor_annihilation = hardening_params[param_map["Annihilation Factor"]];

  for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    RealType const
    generation = factor_generation * std::sqrt(densities_forest[ss_index]);

    RealType const
    annihilation = factor_annihilation * state_hardening[ss_index];

    auto const
    ss_index_global = slip_family.slip_system_indices_[ss_index];

    state_hardening_np1[ss_index_global] = state_hardening[ss_index];

    if (generation > annihilation)
    {
    state_hardening_np1[ss_index_global] += 
        dt * (generation - annihilation) * std::abs(rate_slip[ss_index_global]);
    }

    // Compute the slip resistance
    slip_resistance[ss_index_global] = 
        slip_family.factor_geometry_dislocation_ * slip_family.modulus_shear_ * slip_family.magnitude_burgers_ *
        std::sqrt(densities_parallel[ss_index]);
  }
}

//
// No hardening
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
void
CP::NoHardeningParameters<NumDimT, NumSlipT>::
createLatentMatrix(
  CP::SlipFamily<NumDimT, NumSlipT> & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems)
{
  slip_family.latent_matrix_.set_dimension(slip_family.num_slip_sys_);
  slip_family.latent_matrix_.fill(Intrepid2::ZEROS);

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
void
CP::NoHardeningLaw<NumDimT, NumSlipT, ScalarT>::
harden(
  CP::SlipFamily<NumDimT, NumSlipT> const & slip_family, 
  std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
  RealType dt,
  Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance)
{  
  auto const
  num_slip_sys = slip_family.num_slip_sys_;

  auto const
  slip_system_indices = slip_family.slip_system_indices_;

  for (int ss_index(0); ss_index < num_slip_sys; ++ss_index)
  {
    auto const
    ss_index_global = slip_system_indices[ss_index];

    state_hardening_np1[ss_index_global] = state_hardening_n[ss_index_global];
    slip_resistance[ss_index_global] = state_hardening_np1[ss_index_global];
  }
}
