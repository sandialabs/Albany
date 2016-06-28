//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>

//
//! Convert Euler (Bunge) angles to basis vector
//
template<typename ArgT>
void
CP::eulerAnglesToBasisVectors(ArgT euler_phi_1,
			      ArgT euler_Phi,
			      ArgT euler_phi_2,
			      std::vector<ArgT>& basis_1,
			      std::vector<ArgT>& basis_2,
			      std::vector<ArgT>& basis_3)
{
  using std::sin;
  using std::cos;

  ArgT R[][3] = {{0.0, 0.0, 0.0},
		 {0.0, 0.0, 0.0},
		 {0.0, 0.0, 0.0}};

  // Active rotation tensor
  R[0][0] =  cos(euler_phi_1)*cos(euler_phi_2) - sin(euler_phi_1)*sin(euler_phi_2)*cos(euler_Phi);
  R[0][1] = -cos(euler_phi_1)*sin(euler_phi_2) - sin(euler_phi_1)*cos(euler_phi_2)*cos(euler_Phi);
  R[0][2] =  sin(euler_phi_1)*sin(euler_Phi);
  R[1][0] =  sin(euler_phi_1)*cos(euler_phi_2) + cos(euler_phi_1)*sin(euler_phi_2)*cos(euler_Phi);
  R[1][1] = -sin(euler_phi_1)*sin(euler_phi_2) + cos(euler_phi_1)*cos(euler_phi_2)*cos(euler_Phi);
  R[1][2] = -cos(euler_phi_1)*sin(euler_Phi);
  R[2][0] =  sin(euler_phi_2)*sin(euler_Phi);
  R[2][1] =  cos(euler_phi_2)*sin(euler_Phi);
  R[2][2] =  cos(euler_Phi);

  ArgT e1[3] = {1.0, 0.0, 0.0};
  ArgT e2[3] = {0.0, 1.0, 0.0};
  ArgT e3[3] = {0.0, 0.0, 1.0};

  basis_1 = std::vector<ArgT>(3, 0.0);
  basis_2 = std::vector<ArgT>(3, 0.0);
  basis_3 = std::vector<ArgT>(3, 0.0);
   
  for (int i=0 ; i<3 ; i++) {
    for (int j=0 ; j<3 ; j++) {
      basis_1[i] += R[i][j]*e1[j];
      basis_2[i] += R[i][j]*e2[j];
      basis_3[i] += R[i][j]*e3[j];
    }
  }
}

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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::applySlipIncrement(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp_n,
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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  Intrepid2::Index const
  num_slip = rate_slip.get_dimension();

  if(num_slip == 0) {
    return;
  }

  using HARDENING = CP::HardeningBase<NumDimT, NumSlipT, DataT, ArgT>;

  std::unique_ptr<HARDENING>
  phardening = 
    CP::hardeningFactory<NumDimT, NumSlipT, DataT, ArgT>(
      slip_systems[0].hardening_law);

  HARDENING &
  hardening = *phardening;
 
  hardening.createLatentMatrix(slip_systems);
  hardening.harden(
      slip_systems, 
      dt, 
      rate_slip, 
      state_hardening_n, 
      state_hardening_np1, 
      slip_resistance);

  return;

}




///
/// Update the plastic slips
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::updateSlip(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_resistance,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1)
{
  Intrepid2::Index const num_slip_sys = slip_n.get_dimension();

  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys) {

    // Material properties
    DataT const
    tauC = slip_systems[slip_sys].tau_critical_;

    DataT const
    m = slip_systems[slip_sys].exponent_rate_;

    DataT const
    g0 = slip_systems[slip_sys].rate_slip_reference_;

    // Compute slip increment
    ArgT const
    temp = shear[slip_sys] / slip_resistance[slip_sys];

    slip_np1[slip_sys] = slip_n[slip_sys] + 
      dt * g0 * std::pow(std::fabs(temp), m-1) * temp;

  }

}



///
/// Compute the stresses 
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT, typename DataS>
void
CP::computeStress(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid2::Tensor4<DataS, NumDimT> const & C,
    Intrepid2::Tensor<DataS, NumDimT> const & F,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp,
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
      "Cauchy stress in CrystalPlasticityNLS::computeStress()");

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
// Define nonlinear system based on residual of slip values
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::CrystalPlasticityNLS(
      Intrepid2::Tensor4<ArgT, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C),
      slip_systems_(slip_systems),
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
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
  CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, ArgT>::value(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid2::Vector<T, N> const & x)
{
  // Get a convenience reference to the failed flag in case it is used more
  // than once.
  bool &
  failed = Intrepid2::Function_Base<
  CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, ArgT>::failed;

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

  RealType
  norm_slip_residual_;

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
  CP::applySlipIncrement<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1);

  // Compute state_hardening_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_np1,
      slip_resistance);

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT>(
      slip_systems_,
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
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::hessian(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
      CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, ArgT>::hessian(
      *this,
      x);
}


//
// Define nonlinear system based on residual of slip and hardness values
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::ResidualSlipHardnessNLS(
      Intrepid2::Tensor4<ArgT, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C),
      slip_systems_(slip_systems),
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
  ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, ArgT>::value(
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
  ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, ArgT>::failed;

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

  RealType
  norm_slip_residual_;

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
  CP::applySlipIncrement<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);
  
  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT>(
      slip_systems_,
      C_peeled,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1);

  // Compute state_hardening_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      rate_slip,
      state_hardening_n_,
      state_hardening_computed,
      slip_resistance);

  // Compute slips
  CP::updateSlip<NumDimT, NumSlipT>(
      slip_systems_,
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
      ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, ArgT>::hessian(
      *this,
      x);
}

//
// Linear hardening with recovery
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::LinearMinusRecoveryHardening<NumDimT, NumSlipT, DataT, ArgT>::
createLatentMatrix(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems)
{
  Intrepid2::Index const
  num_slip_sys = slip_systems.size();

  latent_matrix.set_dimension(num_slip_sys);
  latent_matrix.fill(Intrepid2::ONES);

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::LinearMinusRecoveryHardening<NumDimT, NumSlipT, DataT, ArgT>::
harden(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
  DataT dt,
  Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  Intrepid2::Index const
  num_slip_sys = state_hardening_n.get_dimension();

  Intrepid2::Vector<ArgT, NumSlipT>
  rate_slip_abs(num_slip_sys);

  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys) {
    rate_slip_abs(slip_sys) = std::fabs(rate_slip(slip_sys));
  }

  Intrepid2::Vector<ArgT, NumSlipT> const 
  driver_hardening = latent_matrix * rate_slip_abs;

  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys)
  {
    RealType const
    H = slip_systems[slip_sys].H_;

    RealType const
    Rd = slip_systems[slip_sys].Rd_; 

    RealType const
    stress_critical = slip_systems[slip_sys].tau_critical_;

    if (Rd > 0.0) {
      RealType const 
      effective_slip_n = -1.0/Rd * std::log(1.0 - Rd/H * state_hardening_n[slip_sys]);

      state_hardening_np1[slip_sys] = H / Rd * (1.0 - 
        std::exp(-Rd * (effective_slip_n + dt * driver_hardening[slip_sys])));  
    }
    else {
      state_hardening_np1[slip_sys] = 
        state_hardening_n[slip_sys] + H * dt * driver_hardening[slip_sys];
    }

//    state_hardening_np1[slip_sys] = state_hardening_n[slip_sys] +
//      dt * (H - Rd * (state_hardening_n[slip_sys])) * driver_hardening[slip_sys];

    slip_resistance[slip_sys] = stress_critical + state_hardening_np1[slip_sys];
  } 

  return;
}

//
// Saturation hardening
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::SaturationHardening<NumDimT, NumSlipT, DataT, ArgT>::
createLatentMatrix(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems)
{
  Intrepid2::Index const
  num_slip_sys = slip_systems.size();

  latent_matrix.set_dimension(num_slip_sys);
  latent_matrix.fill(Intrepid2::ZEROS);

  for (int slip_sys_i(0); slip_sys_i < num_slip_sys; ++slip_sys_i) {

    for (int slip_sys_j(0); slip_sys_j < num_slip_sys; ++slip_sys_j) {

      latent_matrix(slip_sys_i, slip_sys_j) = 
        std::fabs(Intrepid2::dotdot(
          Intrepid2::sym(slip_systems[slip_sys_i].projector_),
          Intrepid2::sym(slip_systems[slip_sys_j].projector_)));

    }

  }

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::SaturationHardening<NumDimT, NumSlipT, DataT, ArgT>::
harden(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
  DataT dt,
  Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  Intrepid2::Index const
  num_slip_sys = state_hardening_n.get_dimension();

  Intrepid2::Vector<ArgT, NumSlipT>
  rate_slip_abs(num_slip_sys);

  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys) {
    rate_slip_abs(slip_sys) = std::fabs(rate_slip(slip_sys));
  }

  Intrepid2::Vector<ArgT, NumSlipT> const 
  driver_hardening = latent_matrix * rate_slip_abs;

  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys)
  {
    DataT const
    rate_slip_reference = slip_systems[slip_sys].rate_slip_reference_;

    DataT const
    stress_saturation_initial = slip_systems[slip_sys].stress_saturation_initial_;

    DataT const
    rate_hardening = slip_systems[slip_sys].rate_hardening_;

    DataT const
    resistance_slip_initial = slip_systems[slip_sys].resistance_slip_initial_;

    DataT const
    exponent_saturation = slip_systems[slip_sys].exponent_saturation_;  

    ArgT
    effective_slip_rate{0.0};

    // calculate effective slip increment
    for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys) {
      effective_slip_rate += fabs(rate_slip[slip_sys]);
    }

    ArgT
    stress_saturation{stress_saturation_initial};

    if (exponent_saturation > 0.0) {
      stress_saturation = stress_saturation_initial * std::pow(
        effective_slip_rate / rate_slip_reference, exponent_saturation);
    }

    state_hardening_np1[slip_sys] = state_hardening_n[slip_sys] +
      dt * rate_hardening * driver_hardening[slip_sys] *
      (stress_saturation - state_hardening_n[slip_sys]) / 
      (stress_saturation - resistance_slip_initial);

    slip_resistance[slip_sys] = state_hardening_np1[slip_sys];
  } 

  return;
}

//
// Dislocation-density based hardening
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::DislocationDensityHardening<NumDimT, NumSlipT, DataT, ArgT>::
createLatentMatrix(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems)
{
  Intrepid2::Index const
  num_dim = slip_systems[0].s_.get_dimension();

  Intrepid2::Index const
  num_slip_sys = slip_systems.size();

  latent_matrix.set_dimension(num_slip_sys);
  latent_matrix.fill(Intrepid2::ZEROS);

  // std::cout << "latent_matrix" << std::endl;

  for (int slip_sys_i(0); slip_sys_i < num_slip_sys; ++slip_sys_i)
  {
    Intrepid2::Vector<RealType, CP::MAX_DIM>
    normal_i(num_dim);

    normal_i = slip_systems[slip_sys_i].n_;

    for (int slip_sys_j(0); slip_sys_j < num_slip_sys; ++slip_sys_j)
    {
      Intrepid2::Vector<RealType, CP::MAX_DIM>
      direction_j(num_dim);

      direction_j = slip_systems[slip_sys_j].s_;

      Intrepid2::Vector<RealType, CP::MAX_DIM>
      normal_j(num_dim);

      normal_j = slip_systems[slip_sys_j].n_;

      Intrepid2::Vector<RealType, CP::MAX_DIM>
      transverse_j(num_dim);

      transverse_j = Intrepid2::unit(Intrepid2::cross(normal_j, direction_j));

      latent_matrix(slip_sys_i, slip_sys_j) = 
          std::abs(Intrepid2::dot(normal_i, transverse_j));

      // std::cout << std::setprecision(3) << latent_matrix(slip_sys_i, slip_sys_j) << " ";
    }
    // std::cout << std::endl;
  }

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::DislocationDensityHardening<NumDimT, NumSlipT, DataT, ArgT>::
harden(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
  DataT dt,
  Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  Intrepid2::Index const
  num_slip_sys = state_hardening_n.get_dimension();

  //
  // Compute the effective dislocation density at step n
  //
  RealType
  sum_densities_n{0.0};

  Intrepid2::Vector<DataT, NumSlipT>
  densities_forest(num_slip_sys);

  densities_forest = latent_matrix * state_hardening_n;

  // std::cout << "densities_forest: " << densities_forest << std::endl;

  Intrepid2::Tensor<DataT, NumSlipT>
  aux_matrix(num_slip_sys);

  for (int slip_sys_i(0); slip_sys_i < num_slip_sys; ++slip_sys_i)
  {
    for (int slip_sys_j(0); slip_sys_j < num_slip_sys; ++slip_sys_j)
    {
      aux_matrix(slip_sys_i, slip_sys_j) =
          std::sqrt(1.0 - std::pow(latent_matrix(slip_sys_i, slip_sys_j), 2));
    }
  }

  Intrepid2::Vector<RealType, CP::MAX_SLIP>
  densities_parallel(num_slip_sys);

  densities_parallel = aux_matrix * state_hardening_n;

  // std::cout << "densities_parallel: " << densities_parallel << std::endl;

  //
  // Update dislocation densities
  //
  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys)
  {
    DataT const
    generation = 
        slip_systems[slip_sys].c_generation_ * 
        std::sqrt(densities_forest[slip_sys]);

    DataT const
    annihilation =
        slip_systems[slip_sys].c_annihilation_ * state_hardening_n[slip_sys];

    state_hardening_np1[slip_sys] = state_hardening_n[slip_sys];

    if (generation > annihilation)
    {
    state_hardening_np1[slip_sys] += 
        dt * (generation - annihilation) * std::abs(rate_slip[slip_sys]);
    }

    // std::cout << "state_hardening_np1_" << slip_sys << ": ";
    // std::cout << state_hardening_np1[slip_sys] << std::endl;
  }
  
  //
  // Compute the slip resistance
  //
  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys)
  {
    slip_resistance[slip_sys] = 
        slip_systems[slip_sys].factor_geometry_dislocation_ *
        slip_systems[slip_sys].modulus_shear_ * 
        slip_systems[slip_sys].magnitude_burgers_ *
        std::sqrt(densities_parallel[slip_sys]);

    // std::cout << "slip_resistance_" << slip_sys << ": " << slip_resistance[slip_sys] << std::endl;
  }
}

//
// No hardening
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::NoHardening<NumDimT, NumSlipT, DataT, ArgT>::
createLatentMatrix(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems)
{
  Intrepid2::Index const
  num_slip_sys = slip_systems.size();

  latent_matrix.set_dimension(num_slip_sys);
  latent_matrix.fill(Intrepid2::ZEROS);

  return;
}

//
// 
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
void
CP::NoHardening<NumDimT, NumSlipT, DataT, ArgT>::
harden(
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
  DataT dt,
  Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
  Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
  Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
  Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  Intrepid2::Index const
  num_slip_sys = state_hardening_n.get_dimension();
  
  for (int slip_sys(0); slip_sys < num_slip_sys; ++slip_sys)
  {
    state_hardening_np1[slip_sys] = state_hardening_n[slip_sys];
    slip_resistance[slip_sys] = state_hardening_np1[slip_sys];
  }
}
