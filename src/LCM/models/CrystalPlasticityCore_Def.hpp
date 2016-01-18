//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>

  enum FlowRule
  {
    POWER_LAW = 0, THERMAL_ACTIVATION = 1
  };

  enum HardeningLaw
  {
    EXPONENTIAL = 0, SATURATION = 1
  };

  enum TypeResidual
  {
    SLIP_INCREMENT = 0, SLIP_HARDENING = 1
  };

  

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
  Intrepid2::Index num_slip = slip_n.get_dimension();
  Intrepid2::Index num_dim = Fp_n.get_dimension();

  DataT temp;
  Intrepid2::Tensor<RealType, NumDimT> P;
  P.set_dimension(num_dim);
  Intrepid2::Tensor<ArgT, NumDimT> exp_L_dt;
  exp_L_dt.set_dimension(num_dim);

  Lp_np1.fill(Intrepid2::ZEROS);
  for (int s(0); s < num_slip; ++s) {

    // material parameters
    P = slip_systems[s].projector_;

    // calculate plastic velocity gradient
    if(dt > 0){
    Lp_np1 += (slip_np1[s] - slip_n[s])/dt * P;
    }
  }

  CP::confirmTensorSanity<NumDimT>(Lp_np1, "Lp_np1 in applySlipIncrement().");

  // update plastic deformation gradient
  // F^{p}_{n+1} = exp(L_{n+1} * delta t) F^{p}_{n}
  exp_L_dt = Intrepid2::exp(Lp_np1 * dt);
  Fp_np1 = exp_L_dt * Fp_n;

  CP::confirmTensorSanity<NumDimT>(Fp_np1, "Fp_np1 in applySlipIncrement()");
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & hardness_n,
    Intrepid2::Vector<ArgT, NumSlipT> & hardness_np1)
{
  DataT H, Rd;
  ArgT temp, effective_slip_rate(0.0);
  Intrepid2::Index num_slip = rate_slip.get_dimension();

  // calculate effective slip increment
  for (int iSlipSystem(0); iSlipSystem < num_slip; ++iSlipSystem) {
	  effective_slip_rate += fabs(rate_slip[iSlipSystem]);
  }

  for (int s(0); s < num_slip; ++s) {

    // 
    if (slip_systems[s].hardening_law == EXPONENTIAL) {

      DataT H, Rd;
      ArgT effective_slip_n(0.0);
      H = slip_systems[s].H_;
      Rd = slip_systems[s].Rd_;

      hardness_np1[s] = hardness_n[s];

      // calculate additional hardening
      //
      // total hardness = tauC + hardness_np1[s]
      // function form is hardening minus recovery, H/Rd*(1 - exp(-Rd*effective_slip))
      // for reference, another flavor is A*(1 - exp(-B/A*effective_slip)) where H = B and Rd = B/A
      // if H is not specified, H = 0.0, if Rd is not specified, Rd = 0.0
 
      if (Rd > 0.0) {
        //
        effective_slip_n = -1.0/Rd * std::log(1.0 - Rd/H * hardness_n[s]);
        hardness_np1[s] = H / Rd * (1.0 - 
                          std::exp(-Rd * (effective_slip_n + dt * effective_slip_rate)));  
      }
      else {
        hardness_np1[s] = hardness_n[s] + H * dt * effective_slip_rate;
      }

    } 
    //
    else if (slip_systems[s].hardening_law == SATURATION) {

      DataT stress_saturation_initial, rate_hardening, resistance_slip_initial,
        exponent_saturation, rate_slip_reference;

      ArgT driver_hardening, stress_saturation;

      Intrepid2::Index num_slip = rate_slip.get_dimension();

      for (int iSlipSystem(0); iSlipSystem < num_slip; ++iSlipSystem) {

          // material parameters
        rate_slip_reference = slip_systems[iSlipSystem].rate_slip_reference_;
        stress_saturation_initial = 
          slip_systems[iSlipSystem].stress_saturation_initial_;
        rate_hardening = slip_systems[iSlipSystem].rate_hardening_;
        resistance_slip_initial = 
          slip_systems[iSlipSystem].resistance_slip_initial_;
        exponent_saturation = slip_systems[iSlipSystem].exponent_saturation_;

        if (exponent_saturation == 0.0) {
          stress_saturation = stress_saturation_initial;
        }
        else {
          stress_saturation = stress_saturation_initial * std::pow(
            effective_slip_rate / rate_slip_reference, exponent_saturation);
        }

        driver_hardening = 0.0;

        for (int jSlipSystem(0); jSlipSystem < num_slip; ++jSlipSystem) {

          // TODO: calculate hardening matrix during initialization
          driver_hardening += 0.5 *
              std::fabs
              (
                Intrepid2::dotdot
                (
                  slip_systems[iSlipSystem].projector_ + 
                    Intrepid2::transpose(slip_systems[iSlipSystem].projector_),
                  slip_systems[jSlipSystem].projector_ + 
                    Intrepid2::transpose(slip_systems[jSlipSystem].projector_)
                )
              ) *
              std::fabs(rate_slip[jSlipSystem]);

        }

        // TODO: make hardness_n* equal g rather than g-g0
        hardness_np1[iSlipSystem] = hardness_n[iSlipSystem] +
            dt * rate_hardening *
            (stress_saturation - hardness_n[iSlipSystem] - resistance_slip_initial) / 
              (stress_saturation - resistance_slip_initial) * driver_hardening;

      }

      //std::cout << "Hardening driver " << driver_hardening << std::endl;

    }
    //TODO: Re-implement this when rate_slip is the right size
    // else {
    //   TEUCHOS_TEST_FOR_EXCEPTION(
    //     true,
    //     std::logic_error,
    //     "\n**** Error in CrystalPlasticityModel, invalid hardening law\n");
    // }
  }
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::computeResidual(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Vector<ArgT, NumSlipT> const & hardness_np1,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_residual,
    ArgT & norm_slip_residual)
{
  Intrepid2::Index num_slip = slip_n.get_dimension();

  DataT g0, tauC, m;
  //DataT one_over_m;
  ArgT dgamma_value1, dgamma_value2, temp;
  //ArgT temp2;

  for (int s(0); s < num_slip; ++s) {

    // Material properties
    tauC = slip_systems[s].tau_critical_;
    m = slip_systems[s].exponent_rate_;
    //one_over_m = 1.0/m;

    g0 = slip_systems[s].rate_slip_reference_;

    // The current computed value of dgamma
    dgamma_value1 = slip_np1[s] - slip_n[s];

    // Compute slip increment using Fe_np1
    temp = shear_np1[s] / (tauC + hardness_np1[s]);

    dgamma_value2 = dt * g0 * std::pow(std::fabs(temp), m-1) * temp;

    //The difference between the slip increment calculations is the residual for this slip system
    slip_residual[s] = dgamma_value1 - dgamma_value2;

  }

  // Take norm of residual - protect sqrt (Saccado)
  norm_slip_residual = 0.0;
  for (unsigned int i = 0; i < slip_residual.get_dimension(); ++i) {
    norm_slip_residual += slip_residual[i] * slip_residual[i];
  }
  if (norm_slip_residual > 0.0) {
    norm_slip_residual = std::sqrt(norm_slip_residual);
  }
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::computeStress(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid2::Tensor4<RealType, NumDimT> const & C,
    Intrepid2::Tensor<ArgT, NumDimT> const & F,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear)
{
  Intrepid2::Index num_dim = F.get_dimension();
  Intrepid2::Index num_slip = shear.get_dimension();

  Intrepid2::Tensor<DataT, NumDimT> Fpinv;
  Fpinv.set_dimension(num_dim);
  Intrepid2::Tensor<ArgT, NumDimT> Fe;
  Fe.set_dimension(num_dim);
  Intrepid2::Tensor<ArgT, NumDimT> E;
  E.set_dimension(num_dim);
  Intrepid2::Tensor<ArgT, NumDimT> Ce;
  Ce.set_dimension(num_dim);

  Intrepid2::Tensor<RealType, NumDimT> I;
  I.set_dimension(num_dim);
  I.fill(Intrepid2::ZEROS);
  for (int i = 0; i < num_dim; ++i) {
    I(i, i) = 1.0;
  }

  // Saint Venant–Kirchhoff model
  Fpinv = Intrepid2::inverse(Fp);
  Fe = F * Fpinv;
  Ce = Intrepid2::transpose(Fe) * Fe;
  E = 0.5 * (Ce - I);
  S = Intrepid2::dotdot(C, E);
  sigma = (1.0 / Intrepid2::det(Fe)) * Fe * S * Intrepid2::transpose(Fe);
  CP::confirmTensorSanity<NumDimT>(
      sigma,
      "Cauchy stress in CrystalPlasticityNLS::computeStress()");

  // Compute resolved shear stresses
  for (int s(0); s < num_slip; ++s) {
    shear[s] = Intrepid2::dotdot(slip_systems[s].projector_, Ce * S);
  }
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::updateSlipViaExplicitIntegration(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & hardness,
    Intrepid2::Tensor<ArgT, NumDimT> const & S,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1)
    {
  DataT g0, tauC, m;
  ArgT temp;
  Intrepid2::Index num_slip = hardness.get_dimension();

  for (int s(0); s < num_slip; ++s) {

    tauC = slip_systems[s].tau_critical_;
    m = slip_systems[s].exponent_rate_;
    g0 = slip_systems[s].rate_slip_reference_;

    temp = shear[s] / (tauC + hardness[s]);
    slip_np1[s] = slip_n[s] + dt * g0 * std::pow(std::fabs(temp), m-1) * temp;
  }
}



//
// Define nonlinear system based on residual of slip values
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::CrystalPlasticityNLS(
      Intrepid2::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C), slip_systems_(slip_systems), Fp_n_(Fp_n), hardness_n_(hardness_n),
      slip_n_(slip_n),
      F_np1_(F_np1), dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = hardness_n_.get_dimension();
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
    Intrepid2::Vector<T, N> const & x) const
{
  // Tensor mechanical state variables
  Intrepid2::Tensor<T, NumDimT> Fp_np1(num_dim_);
  Intrepid2::Tensor<T, NumDimT> Lp_np1(num_dim_);
  Intrepid2::Tensor<T, NumDimT> sigma_np1(num_dim_);
  Intrepid2::Tensor<T, NumDimT> S_np1(num_dim_);

  // Slip system state variables
  Intrepid2::Vector<T, NumSlipT> hardness_np1(num_slip_);
  Intrepid2::Vector<T, NumSlipT> slip_np1(num_slip_);
  Intrepid2::Vector<T, NumSlipT> shear_np1(num_slip_);
  Intrepid2::Vector<T, NumSlipT> slip_residual(num_slip_);
  Intrepid2::Vector<T, NumSlipT> rate_slip(num_slip_);
  Intrepid2::Vector<T, N> residual;
  T norm_slip_residual_;

  auto const
  num_unknowns = x.get_dimension();

  residual.set_dimension(num_unknowns);

  Intrepid2::Tensor<T, NumDimT>
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
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

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rate_slip.fill(Intrepid2::ZEROS);
  }

  // Compute hardness_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      rate_slip,
      hardness_n_,
      hardness_np1);

  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT>(
      slip_systems_,
      C_,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1);

  // Compute slip_residual and norm_slip_residual
  CP::computeResidual<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      slip_n_, 
      slip_np1,
      hardness_np1,
      shear_np1,
      slip_residual,
      norm_slip_residual_);

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_residual[i];
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
      Intrepid2::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C), slip_systems_(slip_systems), Fp_n_(Fp_n), hardness_n_(hardness_n),
      slip_n_(slip_n),
      F_np1_(F_np1), dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = hardness_n_.get_dimension();
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
    Intrepid2::Vector<T, N> const & x) const
{
  // Tensor mechanical state variables
  Intrepid2::Tensor<T, NumDimT> Fp_np1(num_dim_);
  Intrepid2::Tensor<T, NumDimT> Lp_np1(num_dim_);
  Intrepid2::Tensor<T, NumDimT> sigma_np1(num_dim_);
  Intrepid2::Tensor<T, NumDimT> S_np1(num_dim_);

  // Slip system state variables
  Intrepid2::Vector<T, NumSlipT> hardness_np1(num_slip_);
  Intrepid2::Vector<T, NumSlipT> hardness_computed(num_slip_);
  Intrepid2::Vector<T, NumSlipT> hardness_residual(num_slip_);
  Intrepid2::Vector<T, NumSlipT> slip_np1(num_slip_);
  Intrepid2::Vector<T, NumSlipT> shear_np1(num_slip_);
  Intrepid2::Vector<T, NumSlipT> slip_residual(num_slip_);
  Intrepid2::Vector<T, NumSlipT> rate_slip(num_slip_);
  T norm_slip_residual_;
  Intrepid2::Vector<T, N> residual;

  auto const
  num_unknowns = x.get_dimension();

  residual.set_dimension(num_unknowns);

  Intrepid2::Tensor<T, NumDimT>
  F_np1_peeled = LCM::peel_tensor<EvalT, T, N, NumDimT>()(F_np1_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
    hardness_np1[i] = x[i + num_slip_];
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

  if(dt_ > 0.0){
    rate_slip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rate_slip.fill(Intrepid2::ZEROS);
  }

  // Compute hardness_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      rate_slip,
      hardness_n_,
      hardness_computed);

  for (int i = 0; i< num_slip_; ++i){
    hardness_residual[i] = hardness_np1[i] - hardness_computed[i];
  }
  
  // Compute sigma_np1, S_np1, and shear_np1
  CP::computeStress<NumDimT, NumSlipT>(
      slip_systems_,
      C_,
      F_np1_peeled,
      Fp_np1,
      sigma_np1,
      S_np1,
      shear_np1);

  // Compute slip_residual and norm_slip_residual
  CP::computeResidual<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      slip_n_, 
      slip_np1,
      hardness_np1,
      shear_np1,
      slip_residual,
      norm_slip_residual_);

  for (int i = 0; i< num_slip_; ++i){
    residual[i] = slip_residual[i];
    residual[i + num_slip_] = hardness_residual[i];
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
