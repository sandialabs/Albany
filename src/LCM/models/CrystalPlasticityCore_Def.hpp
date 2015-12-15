//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>

template<Intrepid::Index NumDimT, typename ArgT>
void
LCM::CP::confirmTensorSanity(
    Intrepid::Tensor<ArgT, NumDimT> const & input,
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

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT,
    typename ArgT>
void
LCM::CP::applySlipIncrement(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
    Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid::Tensor<ScalarT, NumDimT> const & Fp_n,
    Intrepid::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid::Tensor<ArgT, NumDimT> & Fp_np1)
{
  Intrepid::Index num_slip = slip_n.get_dimension();
  Intrepid::Index num_dim = Fp_n.get_dimension();

  ScalarT temp;
  Intrepid::Tensor<RealType, NumDimT> P;
  P.set_dimension(num_dim);
  Intrepid::Tensor<ArgT, NumDimT> expL;
  expL.set_dimension(num_dim);

  Lp_np1.fill(Intrepid::ZEROS);
  for (int s(0); s < num_slip; ++s) {

    // material parameters
    P = slip_systems[s].projector_;

    // calculate plastic velocity gradient
    Lp_np1 += (slip_np1[s] - slip_n[s]) * P;
  }

  CP::confirmTensorSanity<NumDimT>(Lp_np1, "Lp_np1 in applySlipIncrement().");

  // update plastic deformation gradient

  //std::cout  << "Lp_np1 " << Lp_np1;

  expL = Intrepid::exp(Lp_np1);
  Fp_np1 = expL * Fp_n;

  CP::confirmTensorSanity<NumDimT>(Fp_np1, "Fp_np1 in applySlipIncrement()");
}

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT,
    typename ArgT>
void
LCM::CP::updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid::Vector<ScalarT, NumSlipT> const & hardness_n,
    Intrepid::Vector<ArgT, NumSlipT> & hardness_np1)
{
  ScalarT H, Rd;
  ArgT temp, slipEffective(0.0);
  Intrepid::Index num_slip = slip_np1.get_dimension();

  for (int iSlipSystem(0); iSlipSystem < num_slip; ++iSlipSystem) {
	  slipEffective += fabs(slip_np1[iSlipSystem]);
  }

  for (int s(0); s < num_slip; ++s) {

    // material parameters
    H = slip_systems[s].H_;
    Rd = slip_systems[s].Rd_;

    hardness_np1[s] = hardness_n[s];

    // calculate additional hardening
    //
    // total hardness = tauC + hardness_np1[s]
    // TODO: tauC -> tau0. This is a bit confusing.
    // function form is hardening minus recovery, H/Rd*(1 - exp(-Rd*eqps))
    // for reference, another flavor is A*(1 - exp(-B/A*eqps)) where H = B and Rd = B/A
    // if H is not specified, H = 0.0, if Rd is not specified, Rd = 0.0
    if (Rd > 0.0) {
      temp = H / Rd * (1.0 - std::exp(-Rd * slipEffective));
    }
    else {
      temp = H * slipEffective;
    }
    // only evolve if hardness increases
    if (temp > hardness_np1[s]) {
      hardness_np1[s] = temp;
    }
  }
}

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT,
    typename ArgT>
void
LCM::CP::computeResidual(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    ScalarT dt,
    Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
    Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid::Vector<ArgT, NumSlipT> const & hardness_np1,
    Intrepid::Vector<ArgT, NumSlipT> const & shear_np1,
    Intrepid::Vector<ArgT, NumSlipT> & slip_residual,
    ArgT & norm_slip_residual)
{
  Intrepid::Index num_slip = slip_n.get_dimension();

  ScalarT g0, tauC, m;
  //ScalarT one_over_m;
  ArgT dgamma_value1, dgamma_value2, temp;
  //ArgT temp2;

  for (int s(0); s < num_slip; ++s) {

    // Material properties
    tauC = slip_systems[s].tau_critical_;
    m = slip_systems[s].gamma_exp_;
    //one_over_m = 1.0/m;

    g0 = slip_systems[s].gamma_dot_0_;

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

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT,
    typename ArgT>
void
LCM::CP::computeStress(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid::Tensor4<RealType, NumDimT> const & C,
    Intrepid::Tensor<ScalarT, NumDimT> const & F,
    Intrepid::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid::Tensor<ArgT, NumDimT> & sigma,
    Intrepid::Tensor<ArgT, NumDimT> & S,
    Intrepid::Vector<ArgT, NumSlipT> & shear)
{
  Intrepid::Index num_dim = F.get_dimension();
  Intrepid::Index num_slip = shear.get_dimension();

  Intrepid::Tensor<ArgT, NumDimT> Fpinv;
  Fpinv.set_dimension(num_dim);
  Intrepid::Tensor<ArgT, NumDimT> Fe;
  Fe.set_dimension(num_dim);
  Intrepid::Tensor<ArgT, NumDimT> E;
  E.set_dimension(num_dim);
  Intrepid::Tensor<ArgT, NumDimT> Ce;
  Ce.set_dimension(num_dim);

  Intrepid::Tensor<RealType, NumDimT> I;
  I.set_dimension(num_dim);
  I.fill(Intrepid::ZEROS);
  for (int i = 0; i < num_dim; ++i) {
    I(i, i) = 1.0;
  }

  // Saint Venant–Kirchhoff model
  Fpinv = Intrepid::inverse(Fp);
  Fe = F * Fpinv;
  Ce = Intrepid::transpose(Fe) * Fe;
  E = 0.5 * (Ce - I);
  S = Intrepid::dotdot(C, E);
  sigma = (1.0 / Intrepid::det(Fe)) * Fe * S * Intrepid::transpose(Fe);
  CP::confirmTensorSanity<NumDimT>(
      sigma,
      "Cauchy stress in CrystalPlasticityNLS::computeStress()");

  // Compute resolved shear stresses
  for (int s(0); s < num_slip; ++s) {
    shear[s] = Intrepid::dotdot(slip_systems[s].projector_, Ce * S);
  }
}

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT,
    typename ArgT>
void
LCM::CP::updateSlipViaExplicitIntegration(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    ScalarT dt,
    Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
    Intrepid::Vector<ScalarT, NumSlipT> const & hardness,
    Intrepid::Tensor<ArgT, NumDimT> const & S,
    Intrepid::Vector<ArgT, NumSlipT> const & shear,
    Intrepid::Vector<ArgT, NumSlipT> & slip_np1)
    {
  ScalarT g0, tauC, m, temp;
  Intrepid::Index num_slip = hardness.get_dimension();

  for (int s(0); s < num_slip; ++s) {

    tauC = slip_systems[s].tau_critical_;
    m = slip_systems[s].gamma_exp_;
    g0 = slip_systems[s].gamma_dot_0_;

    temp = shear[s] / (tauC + hardness[s]);
    slip_np1[s] = slip_n[s] + dt * g0 * std::pow(std::fabs(temp), m-1) * temp;
  }
}

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename EvalT>
LCM::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::CrystalPlasticityNLS(
      Intrepid::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid::Tensor<ScalarT, NumDimT> const & F_np1,
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

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename EvalT>
template<typename T, Intrepid::Index N>
T
LCM::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::value(
    Intrepid::Vector<T, N> const & x)
{
  return Intrepid::Function_Base<
  CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::value(
      *this,
      x);
}

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename EvalT>
template<typename T, Intrepid::Index N>
Intrepid::Vector<T, N>
LCM::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::gradient(
    Intrepid::Vector<T, N> const & slip_np1) const
{
  // DJL todo: Experiment with how/where these are allocated.
  Intrepid::Tensor<T, NumDimT> Fp_np1;
  Intrepid::Tensor<T, NumDimT> Lp_np1;
  Intrepid::Vector<T, N> hardness_np1;
  Intrepid::Tensor<T, NumDimT> sigma_np1;
  Intrepid::Tensor<T, NumDimT> S_np1;
  Intrepid::Vector<T, N> shear_np1;
  Intrepid::Vector<T, N> slip_residual;
  T norm_slip_residual_;

  Fp_np1.set_dimension(num_dim_);
  Lp_np1.set_dimension(num_dim_);
  hardness_np1.set_dimension(num_slip_);
  sigma_np1.set_dimension(num_dim_);
  S_np1.set_dimension(num_dim_);
  shear_np1.set_dimension(num_slip_);
  slip_residual.set_dimension(num_slip_);

  Intrepid::Tensor<T, NumDimT> F_np1_peeled;
  F_np1_peeled.set_dimension(num_dim_);
  for (int i = 0; i < num_dim_; ++i) {
    for (int j = 0; j < num_dim_; ++j) {
      F_np1_peeled(i, j) = peel<EvalT, T, N>()(F_np1_(i, j));
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

  // Compute hardness_np1
  CP::updateHardness<NumDimT, NumSlipT>(
      slip_systems_,
      slip_np1,
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

template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename EvalT>
template<typename T, Intrepid::Index N>
Intrepid::Tensor<T, N>
LCM::CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>::hessian(
    Intrepid::Vector<T, N> const & x)
{
  return Intrepid::Function_Base<
      CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, ScalarT>::hessian(
      *this,
      x);
}
