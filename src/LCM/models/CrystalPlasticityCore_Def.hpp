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
      if (!boost::math::isfinite(
          Sacado::ScalarValue<ArgT>::eval(input(i, j)))) {
        std::string msg =
            "**** Invalid data detected in CP::confirmTensorSanity(): "
                + message;
        TEUCHOS_TEST_FOR_EXCEPTION(
            !boost::math::isfinite(
                Sacado::ScalarValue<ArgT>::eval(input(i, j))),
            std::logic_error,
            msg);
      }
    }
  }
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::applySlipIncrement(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
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
  Intrepid2::Tensor<ArgT, NumDimT> expL;
  expL.set_dimension(num_dim);

  Lp_np1.fill(Intrepid2::ZEROS);
  for (int s(0); s < num_slip; ++s) {

    // material parameters
    P = slip_systems[s].projector_;

    // calculate plastic velocity gradient
    Lp_np1 += (slip_np1[s] - slip_n[s]) * P;
  }

  CP::confirmTensorSanity<NumDimT>(Lp_np1, "Lp_np1 in applySlipIncrement().");

  // update plastic deformation gradient

  //std::cout  << "Lp_np1 " << Lp_np1;

  expL = Intrepid2::exp(Lp_np1);
  Fp_np1 = expL * Fp_n;

  CP::confirmTensorSanity<NumDimT>(Fp_np1, "Fp_np1 in applySlipIncrement()");
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
CP::updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rateSlip,
    Intrepid2::Vector<DataT, NumSlipT> const & hardness_n,
    Intrepid2::Vector<ArgT, NumSlipT> & hardness_np1)
{
  DataT H, Rd;
  ArgT temp, effective_slip_rate(0.0);
  Intrepid2::Index num_slip = rateSlip.get_dimension();

  // calculate effective slip increment
  for (int iSlipSystem(0); iSlipSystem < num_slip; ++iSlipSystem) {
	  effective_slip_rate += fabs(rateSlip[iSlipSystem]);
  }

  for (int s(0); s < num_slip; ++s) {

    // 
    if (slip_systems[s].hardeningLaw == EXPONENTIAL) {

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
    else if (slip_systems[s].hardeningLaw == SATURATION) {

      DataT stressSaturationInitial, rateHardening, resistanceSlipInitial,
        exponentSaturation, rateSlipReference;

      ArgT driverHardening, stressSaturation;

      Intrepid2::Index num_slip = rateSlip.get_dimension();

      for (int iSlipSystem(0); iSlipSystem < num_slip; ++iSlipSystem) {

          // material parameters
        rateSlipReference = slip_systems[iSlipSystem].rateSlipReference_;
        stressSaturationInitial = 
          slip_systems[iSlipSystem].stressSaturationInitial_;
        rateHardening = slip_systems[iSlipSystem].rateHardening_;
        resistanceSlipInitial = 
          slip_systems[iSlipSystem].resistanceSlipInitial_;
        exponentSaturation = slip_systems[iSlipSystem].exponentSaturation_;

        if (exponentSaturation == 0.0) {
          stressSaturation = stressSaturationInitial;
        }
        else {
          stressSaturation = stressSaturationInitial * std::pow(
            effective_slip_rate / rateSlipReference, exponentSaturation);
        }

        driverHardening = 0.0;

        for (int jSlipSystem(0); jSlipSystem < num_slip; ++jSlipSystem) {

          // TODO: calculate hardening matrix during initialization
          driverHardening += 0.5 *
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
              std::fabs(rateSlip[jSlipSystem]);

        }

        // TODO: make hardness_n* equal g rather than g-g0
        hardness_np1[iSlipSystem] = hardness_n[iSlipSystem] +
            dt * rateHardening *
            (stressSaturation - hardness_n[iSlipSystem] - resistanceSlipInitial) / 
              (stressSaturation - resistanceSlipInitial) * driverHardening;

      }

      //std::cout << "Hardening driver " << driverHardening << std::endl;

    }
    //TODO: Re-implement this when rateSlip is the right size
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
    m = slip_systems[s].exponentRate_;
    //one_over_m = 1.0/m;

    g0 = slip_systems[s].rateSlipReference_;

    // The current computed value of dgamma
    dgamma_value1 = slip_np1[s] - slip_n[s];

    // Compute slip increment using Fe_np1
    temp = shear_np1[s] / (tauC + hardness_np1[s]);

//     // establishing normalized filter for active slip systems
//     const double active_filter = std::numeric_limits<RealType>::epsilon() * 10.0;
//     if (temp < active_filter) {
//       dgamma_value2 = dt * g0 * 0.0;
//     }
//     else {
//       dgamma_value2 = dt * g0 * std::pow(temp, m) * sign;
//     }

    dgamma_value2 = dt * g0 * std::pow(std::fabs(temp), m-1) * temp;

    //The difference between the slip increment calculations is the residual for this slip system
    slip_residual[s] = dgamma_value1 - dgamma_value2;

    //residual can take two forms - see Steinmann and Stein, CMAME (2006)
    //establishing filter for gamma, 1.0e-4 for now
    //const double gamma_filter = 1.0e-4;
    //if (dgamma_value2 <= gamma_filter) {
    //  slip_residual[s] = dgamma_value2 - dgamma_value1;
    //}
    //else {
    //  int sign = shear_np1[s] < 0 ? -1 : 1;
    //  temp2 = dgamma_value1 / (dt * g0 * sign);
    //  slip_residual[s] = -std::pow(temp2, one_over_m) + temp;
    //}
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
    Intrepid2::Tensor<DataT, NumDimT> const & F,
    Intrepid2::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear)
{
  Intrepid2::Index num_dim = F.get_dimension();
  Intrepid2::Index num_slip = shear.get_dimension();

  Intrepid2::Tensor<ArgT, NumDimT> Fpinv;
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
    Intrepid2::Vector<DataT, NumSlipT> const & hardness,
    Intrepid2::Tensor<ArgT, NumDimT> const & S,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1)
    {
  DataT g0, tauC, m, temp;
  Intrepid2::Index num_slip = hardness.get_dimension();

  for (int s(0); s < num_slip; ++s) {

    tauC = slip_systems[s].tau_critical_;
    m = slip_systems[s].exponentRate_;
    g0 = slip_systems[s].rateSlipReference_;

    temp = shear[s] / (tauC + hardness[s]);
    slip_np1[s] = slip_n[s] + dt * g0 * std::pow(std::fabs(temp), m-1) * temp;
  }
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::CrystalPlasticityNLS(
      Intrepid2::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<DataT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C), slip_systems_(slip_systems), Fp_n_(Fp_n), hardness_n_(hardness_n),
      slip_n_(slip_n),
      F_np1_(F_np1), dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = hardness_n_.get_dimension();
  DIMENSION = num_slip_;
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
T
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
  CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, DataT>::value(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid2::Vector<T, N> const & slip_np1) const
{
  // DJL todo: Experiment with how/where these are allocated.
  Intrepid2::Tensor<T, NumDimT> Fp_np1;
  Intrepid2::Tensor<T, NumDimT> Lp_np1;
  Intrepid2::Vector<T, N> hardness_np1;
  Intrepid2::Tensor<T, NumDimT> sigma_np1;
  Intrepid2::Tensor<T, NumDimT> S_np1;
  Intrepid2::Vector<T, N> shear_np1;
  Intrepid2::Vector<T, N> slip_residual;
  Intrepid2::Vector<T, N> rateSlip;
  T norm_slip_residual_;

  Fp_np1.set_dimension(num_dim_);
  Lp_np1.set_dimension(num_dim_);
  hardness_np1.set_dimension(num_slip_);
  sigma_np1.set_dimension(num_dim_);
  S_np1.set_dimension(num_dim_);
  shear_np1.set_dimension(num_slip_);
  slip_residual.set_dimension(num_slip_);
  rateSlip.set_dimension(num_slip_);

  Intrepid2::Tensor<T, NumDimT> F_np1_peeled;
  F_np1_peeled.set_dimension(num_dim_);
  for (int i = 0; i < num_dim_; ++i) {
    for (int j = 0; j < num_dim_; ++j) {
      F_np1_peeled(i, j) = LCM::peel<EvalT, T, N>()(F_np1_(i, j));
    }
  }

  // Compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT>(
      slip_systems_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);

  if(dt_ > 0.0){
    rateSlip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rateSlip.fill(Intrepid2::ZEROS);
  }

  // Compute hardness_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      rateSlip,
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

  return slip_residual;
}

// Nonlinear system, residual based on slip increments
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Tensor<T, N>
CP::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::hessian(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
      CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, DataT>::hessian(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::ResidualSlipHardnessNLS(
      Intrepid2::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<DataT, NumDimT> const & F_np1,
      RealType dt)
  :
      C_(C), slip_systems_(slip_systems), Fp_n_(Fp_n), hardness_n_(hardness_n),
      slip_n_(slip_n),
      F_np1_(F_np1), dt_(dt)
{
  num_dim_ = Fp_n_.get_dimension();
  num_slip_ = hardness_n_.get_dimension();
  DIMENSION = num_slip_;
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
T
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid2::Vector<T, N> const & x)
{
  return Intrepid2::Function_Base<
  ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, DataT>::value(
      *this,
      x);
}

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
CP::ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid2::Vector<T, N> const & x) const
{
  // DJL todo: Experiment with how/where these are allocated.
  Intrepid2::Tensor<T, NumDimT> Fp_np1;
  Intrepid2::Tensor<T, NumDimT> Lp_np1;
  Intrepid2::Vector<T, NumSlipT> hardness_np1;
  Intrepid2::Vector<T, NumSlipT> hardness_computed;
  Intrepid2::Vector<T, NumSlipT> hardness_residual;
  Intrepid2::Tensor<T, NumDimT> sigma_np1;
  Intrepid2::Tensor<T, NumDimT> S_np1;
  Intrepid2::Vector<T, NumSlipT> slip_np1;
  Intrepid2::Vector<T, NumSlipT> shear_np1;
  Intrepid2::Vector<T, NumSlipT> slip_residual;
  Intrepid2::Vector<T, NumSlipT> rateSlip;
  T norm_slip_residual_;
  Intrepid2::Vector<T, N> residual;

  Fp_np1.set_dimension(num_dim_);
  Lp_np1.set_dimension(num_dim_);
  hardness_np1.set_dimension(num_slip_);
  hardness_computed.set_dimension(num_slip_);
  hardness_residual.set_dimension(num_slip_);
  sigma_np1.set_dimension(num_dim_);
  S_np1.set_dimension(num_dim_);
  slip_np1.set_dimension(num_slip_);
  shear_np1.set_dimension(num_slip_);
  slip_residual.set_dimension(num_slip_);
  rateSlip.set_dimension(num_slip_);

  for (int i = 0; i< num_slip_; ++i){
    slip_np1[i] = x[i];
    hardness_np1[i] = x[i + num_slip_];
  }
  
  Intrepid2::Tensor<T, NumDimT> F_np1_peeled;
  F_np1_peeled.set_dimension(num_dim_);
  for (int i = 0; i < num_dim_; ++i) {
    for (int j = 0; j < num_dim_; ++j) {
      F_np1_peeled(i, j) = LCM::peel<EvalT, T, N>()(F_np1_(i, j));
    }
  }

  // Compute Lp_np1, and Fp_np1
  CP::applySlipIncrement<NumDimT, NumSlipT>(
      slip_systems_,
      slip_n_,
      slip_np1,
      Fp_n_,
      Lp_np1,
      Fp_np1);

  if(dt_ > 0.0){
    rateSlip = (slip_np1 - slip_n_) / dt_;
  }
  else{
    rateSlip.fill(Intrepid2::ZEROS);
  }

  // Compute hardness_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      dt_,
      rateSlip,
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
      ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, DataT>::hessian(
      *this,
      x);
}
