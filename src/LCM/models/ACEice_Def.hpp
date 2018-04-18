//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ACEice.hpp"
#include "Albany_Utils.hpp"
#include "MiniNonlinearSolver.h"

namespace LCM {

template <typename EvalT, typename Traits>
ACEiceMiniKernel<EvalT, Traits>::ACEiceMiniKernel(
    ConstitutiveModel<EvalT, Traits>&    model,
    Teuchos::ParameterList*              p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model)
{
  // Baseline constants
  sat_mod_ = p->get<RealType>("Saturation Modulus", 0.0);
  sat_exp_ = p->get<RealType>("Saturation Exponent", 0.0);

  ice_density_        = p->get<RealType>("ACE Ice Density", 0.0);
  water_density_      = p->get<RealType>("ACE Water Density", 0.0);
  ice_thermal_cond_   = p->get<RealType>("ACE Ice Thermal Conductivity", 0.0);
  water_thermal_cond_ = p->get<RealType>("ACE Water Thermal Conductivity", 0.0);
  ice_heat_capacity_  = p->get<RealType>("ACE Ice Heat Capacity", 0.0);

  ice_saturation_init_  = p->get<RealType>("ACE Ice Initial Saturation", 0.0);
  ice_saturation_max_   = p->get<RealType>("ACE Ice Maximum Saturation", 0.0);
  water_saturation_min_ = p->get<RealType>("ACE Water Minimum Saturation", 0.0);
  porosity_             = p->get<RealType>("ACE Porosity", 0.0);

  // retrieve appropriate field name strings
  std::string const cauchy_string       = field_name_map_["Cauchy_Stress"];
  std::string const Fp_string           = field_name_map_["Fp"];
  std::string const eqps_string         = field_name_map_["eqps"];
  std::string const yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string const source_string       = field_name_map_["Mechanical_Source"];
  std::string const F_string            = field_name_map_["F"];
  std::string const J_string            = field_name_map_["J"];

  // define the dependent fields
  setDependentField(F_string, dl->qp_tensor);
  setDependentField(J_string, dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);
  setDependentField("Hardening Modulus", dl->qp_scalar);
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Yield Strength", dl->qp_scalar);
  setDependentField("Delta Time", dl->workset_scalar);

  // Computed incrementally
  setEvaluatedField("ACE Ice Saturation", dl->qp_scalar);

  // For output/convenience
  setEvaluatedField("ACE Density", dl->qp_scalar);
  setEvaluatedField("ACE Heat Capacity", dl->qp_scalar);
  setEvaluatedField("ACE Thermal Conductivity", dl->qp_scalar);
  //setEvaluatedField("ACE Porosity", dl->qp_scalar);
  setEvaluatedField("ACE Water Saturation", dl->qp_scalar);

  // define the evaluated fields
  setEvaluatedField(cauchy_string, dl->qp_tensor);
  setEvaluatedField(Fp_string, dl->qp_tensor);
  setEvaluatedField(eqps_string, dl->qp_scalar);
  setEvaluatedField(yieldSurface_string, dl->qp_scalar);
  if (have_temperature_ == true) {
    setEvaluatedField(source_string, dl->qp_scalar);
  }

  // define the state variables

  // stress
  addStateVariable(
      cauchy_string,
      dl->qp_tensor,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Cauchy Stress", false));

  // Fp
  addStateVariable(
      Fp_string,
      dl->qp_tensor,
      "identity",
      0.0,
      true,
      p->get<bool>("Output Fp", false));

  // eqps
  addStateVariable(
      eqps_string,
      dl->qp_scalar,
      "scalar",
      0.0,
      true,
      p->get<bool>("Output eqps", false));

  // yield surface
  addStateVariable(
      yieldSurface_string,
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Yield Surface", false));

  // Ice saturation
  addStateVariable(
      "ACE Ice Saturation",
      dl->qp_scalar,
      "scalar",
      1.0,
      false,
      p->get<bool>("Output ACE Ice Saturation", false));

  // Density
  addStateVariable(
      "ACE Density",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Density", false));

  // Heat Capacity
  addStateVariable(
      "ACE Heat Capacity",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Heat Capacity", false));

  // ACE Thermal Conductivity
  addStateVariable(
      "ACE Thermal Conductivity",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Thermal Conductivity", false));

  // ACE Porosity
  /*
  addStateVariable(
      "ACE Porosity",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Porosity", false));
  */

  // ACE Water Saturation
  addStateVariable(
      "ACE Water Saturation",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("ACE Water Saturation", false));

  // mechanical source
  if (have_temperature_ == true) {
    addStateVariable(
        source_string,
        dl->qp_scalar,
        "scalar",
        0.0,
        false,
        p->get<bool>("Output Mechanical Source", false));
  }
}

template <typename EvalT, typename Traits>
void
ACEiceMiniKernel<EvalT, Traits>::init(
    Workset&                 workset,
    FieldMap<const ScalarT>& input_fields,
    FieldMap<ScalarT>&       output_fields)
{
  std::string cauchy_string       = field_name_map_["Cauchy_Stress"];
  std::string Fp_string           = field_name_map_["Fp"];
  std::string eqps_string         = field_name_map_["eqps"];
  std::string yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string source_string       = field_name_map_["Mechanical_Source"];
  std::string F_string            = field_name_map_["F"];
  std::string J_string            = field_name_map_["J"];

  def_grad = *input_fields[F_string];
  J        = *input_fields[J_string];

  elastic_modulus   = *input_fields["Elastic Modulus"];
  hardening_modulus = *input_fields["Hardening Modulus"];
  poissons_ratio    = *input_fields["Poissons Ratio"];
  yield_strength    = *input_fields["Yield Strength"];

  delta_time        = *input_fields["Delta Time"];

  stress    = *output_fields[cauchy_string];
  Fp        = *output_fields[Fp_string];
  eqps      = *output_fields[eqps_string];
  yieldSurf = *output_fields[yieldSurface_string];

  ice_saturation       = *output_fields["ACE Ice Saturation"];
  density              = *output_fields["ACE Density"];
  heat_capacity        = *output_fields["ACE Heat Capacity"];
  thermal_cond         = *output_fields["ACE Thermal Conductivity"];
  water_saturation     = *output_fields["ACE Water Saturation"];

  if (have_temperature_ == true) {
    source = *output_fields[source_string];
  }

  // get State Variables
  Fpold              = (*workset.stateArrayPtr)[Fp_string + "_old"];
  eqpsold            = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Told               = (*workset.stateArrayPtr)["Temperature_old"];
  ice_saturation_old = (*workset.stateArrayPtr)["ACE Ice Saturation_old"];
}

//
// J2 nonlinear system
//
template <typename EvalT, minitensor::Index M = 1>
class J2NLS : public minitensor::
                  Function_Base<J2NLS<EvalT, M>, typename EvalT::ScalarT, M> {
  using S = typename EvalT::ScalarT;

 public:
  J2NLS(
      RealType sat_mod_,
      RealType sat_exp_,
      RealType eqps_old_,
      S const& K,
      S const& smag,
      S const& mubar,
      S const& Y)
      : sat_mod(sat_mod_), sat_exp(sat_exp_), eqps_old(eqps_old_), K_(K),
        smag_(smag), mubar_(mubar), Y_(Y)
  {
  }

  static constexpr char const* const NAME{"J2 NLS"};

  using Base =
      minitensor::Function_Base<J2NLS<EvalT, M>, typename EvalT::ScalarT, M>;

  // Default value.
  template <typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const& x)
  {
    return Base::value(*this, x);
  }

  // Explicit gradient.
  template <typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const& x)
  {
    // Firewalls.
    minitensor::Index const dimension = x.get_dimension();

    ALBANY_EXPECT(dimension == Base::DIMENSION);

    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const K     = peel<EvalT, T, N>()(K_);
    T const smag  = peel<EvalT, T, N>()(smag_);
    T const mubar = peel<EvalT, T, N>()(mubar_);
    T const Y     = peel<EvalT, T, N>()(Y_);

    // This is the actual computation of the gradient.
    minitensor::Vector<T, N> r(dimension);

    T const& X     = x(0);
    T const  alpha = eqps_old + sq23 * X;
    T const  H     = K * alpha + sat_mod * (1.0 - std::exp(-sat_exp * alpha));
    T const  R     = smag - (2.0 * mubar * X + sq23 * (Y + H));

    r(0) = R;

    return r;
  }

  // Default AD hessian.
  template <typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const& x)
  {
    return Base::hessian(*this, x);
  }

  // Constants.
  RealType const sq23{std::sqrt(2.0 / 3.0)};
  RealType const sat_mod{0.0};
  RealType const sat_exp{0.0};
  RealType const eqps_old{0.0};

  // Inputs
  S const& K_;
  S const& smag_;
  S const& mubar_;
  S const& Y_;
};

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
ACEiceMiniKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  constexpr minitensor::Index MAX_DIM{3};

  using Tensor = minitensor::Tensor<ScalarT, MAX_DIM>;

  Tensor       F(num_dims_);
  Tensor const I(minitensor::eye<ScalarT, MAX_DIM>(num_dims_));
  Tensor       sigma(num_dims_);

  ScalarT const E     = elastic_modulus(cell, pt);
  ScalarT const nu    = poissons_ratio(cell, pt);
  ScalarT const kappa = E / (3.0 * (1.0 - 2.0 * nu));
  ScalarT const mu    = E / (2.0 * (1.0 + nu));
  ScalarT const K     = hardening_modulus(cell, pt);
  ScalarT const Y     = yield_strength(cell, pt);
  ScalarT const J1    = J(cell, pt);
  ScalarT const Jm23  = 1.0 / std::cbrt(J1 * J1);

  // fill local tensors
  F.fill(def_grad, cell, pt, 0, 0);

  // Mechanical deformation gradient
  auto Fm = Tensor(F);
  if (have_temperature_) {
    ScalarT dtemp           = temperature_(cell, pt) - ref_temperature_;
    ScalarT thermal_stretch = std::exp(expansion_coeff_ * dtemp);
    Fm /= thermal_stretch;
  }

  Tensor Fpn(num_dims_);

  for (int i{0}; i < num_dims_; ++i) {
    for (int j{0}; j < num_dims_; ++j) {
      Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
    }
  }

  // Deal with non-mechanical values
  water_saturation(cell, pt) = 1.0 - ice_saturation(cell, pt);

  // compute trial state
  Tensor const  Fpinv = minitensor::inverse(Fpn);
  Tensor const  Cpinv = Fpinv * minitensor::transpose(Fpinv);
  Tensor const  be    = Jm23 * Fm * Cpinv * minitensor::transpose(Fm);
  Tensor        s     = mu * minitensor::dev(be);
  ScalarT const mubar = minitensor::trace(be) * mu / (num_dims_);

  // check yield condition
  ScalarT const smag = minitensor::norm(s);
  ScalarT const sq23{std::sqrt(2.0 / 3.0)};
  ScalarT const f =
      smag - sq23 * (Y + K * eqpsold(cell, pt) +
                     sat_mod_ * (1.0 - std::exp(-sat_exp_ * eqpsold(cell, pt))));

  RealType constexpr yield_tolerance = 1.0e-12;

  if (f > yield_tolerance) {
    // Use minimization equivalent to return mapping
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using NLS    = J2NLS<EvalT>;

    constexpr minitensor::Index nls_dim{NLS::DIMENSION};

    using MIN  = minitensor::Minimizer<ValueT, nls_dim>;
    using STEP = minitensor::NewtonStep<NLS, ValueT, nls_dim>;

    MIN  minimizer;
    STEP step;
    NLS  j2nls(sat_mod_, sat_exp_, eqpsold(cell, pt), K, smag, mubar, Y);

    minitensor::Vector<ScalarT, nls_dim> x;

    x(0) = 0.0;

    LCM::MiniSolver<MIN, STEP, NLS, EvalT, nls_dim> mini_solver(
        minimizer, step, j2nls, x);

    ScalarT const alpha = eqpsold(cell, pt) + sq23 * x(0);
    ScalarT const H     = K * alpha + sat_mod_ * (1.0 - exp(-sat_exp_ * alpha));
    ScalarT const dgam  = x(0);

    // plastic direction
    Tensor const N = (1 / smag) * s;

    // update s
    s -= 2 * mubar * dgam * N;

    // update eqps
    eqps(cell, pt) = alpha;

    // mechanical source
    if (have_temperature_ == true && delta_time(0) > 0) {
      source(cell, pt) =
          (sq23 * dgam / delta_time(0) * (Y + H + temperature_(cell, pt))) /
          (density(cell, pt) * heat_capacity(cell, pt));
    }

    // exponential map to get Fpnew
    Tensor const A     = dgam * N;
    Tensor const expA  = minitensor::exp(A);
    Tensor const Fpnew = expA * Fpn;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp(cell, pt, i, j) = Fpnew(i, j);
      }
    }
  } else {
    eqps(cell, pt) = eqpsold(cell, pt);

    if (have_temperature_ == true) source(cell, pt) = 0.0;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp(cell, pt, i, j) = Fpn(i, j);
      }
    }
  }

  // update yield surface
  yieldSurf(cell, pt) = Y + K * eqps(cell, pt) +
                        sat_mod_ * (1. - std::exp(-sat_exp_ * eqps(cell, pt)));

  // compute pressure
  ScalarT const p = 0.5 * kappa * (J(cell, pt) - 1. / (J(cell, pt)));

  // compute stress
  sigma = p * I + s / J(cell, pt);

  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      stress(cell, pt, i, j) = sigma(i, j);
    }
  }
}
}  // namespace LCM
