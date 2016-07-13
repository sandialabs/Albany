//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MiniNonlinearSolver.h"
#include "J2MiniSolver.hpp"

namespace LCM
{

template<typename EvalT, typename Traits>
J2MiniKernel<EvalT, Traits>::
J2MiniKernel(
    ConstitutiveModel<EvalT, Traits> &model,
    Teuchos::ParameterList * p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
    BaseKernel(model),
    sat_mod(p->get<RealType>("Saturation Modulus", 0.0)),
    sat_exp(p->get<RealType>("Saturation Exponent", 0.0))
{
  // retrive appropriate field name strings
  std::string const cauchy_string = field_name_map_["Cauchy_Stress"];
  std::string const Fp_string = field_name_map_["Fp"];
  std::string const eqps_string = field_name_map_["eqps"];
  std::string const yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string const source_string = field_name_map_["Mechanical_Source"];
  std::string const F_string = field_name_map_["F"];
  std::string const J_string = field_name_map_["J"];

  // define the dependent fields
  setDependentField(F_string, dl->qp_tensor);
  setDependentField(J_string, dl->qp_scalar);
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);
  setDependentField("Yield Strength", dl->qp_scalar);
  setDependentField("Hardening Modulus", dl->qp_scalar);
  setDependentField("Delta Time", dl->workset_scalar);

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
  addStateVariable(cauchy_string, dl->qp_tensor, "scalar", 0.0, false,
          p->get<bool>("Output Cauchy Stress", false));
  
  // Fp
  addStateVariable(Fp_string, dl->qp_tensor, "identity", 0.0, true,
          p->get<bool>("Output Fp", false));
  
  // eqps
  addStateVariable(eqps_string, dl->qp_scalar, "scalar", 0.0, true,
          p->get<bool>("Output eqps", false));
  
  // yield surface
  addStateVariable(yieldSurface_string, dl->qp_scalar, "scalar", 0.0, false,
          p->get<bool>("Output Yield Surface", false));
  //
  // mechanical source
  if (have_temperature_ == true) {
    addStateVariable(source_string, dl->qp_scalar, "scalar", 0.0, false,
            p->get<bool>("Output Mechanical Source", false));
  }
}

template<typename EvalT, typename Traits>
void
J2MiniKernel<EvalT, Traits>::
init(Workset &workset,
     FieldMap<ScalarT> &dep_fields,
     FieldMap<ScalarT> &eval_fields)
{
  std::string cauchy_string = field_name_map_["Cauchy_Stress"];
  std::string Fp_string = field_name_map_["Fp"];
  std::string eqps_string = field_name_map_["eqps"];
  std::string yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string source_string = field_name_map_["Mechanical_Source"];
  std::string F_string = field_name_map_["F"];
  std::string J_string = field_name_map_["J"];

  // extract dependent MDFields
  def_grad = *dep_fields[F_string];
  J = *dep_fields[J_string];
  poissons_ratio = *dep_fields["Poissons Ratio"];
  elastic_modulus = *dep_fields["Elastic Modulus"];
  yieldStrength = *dep_fields["Yield Strength"];
  hardeningModulus = *dep_fields["Hardening Modulus"];
  delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  stress = *eval_fields[cauchy_string];
  Fp = *eval_fields[Fp_string];
  eqps = *eval_fields[eqps_string];
  yieldSurf = *eval_fields[yieldSurface_string];

  if (have_temperature_ == true) {
    source = *eval_fields[source_string];
  }

  // get State Variables
  Fpold = (*workset.stateArrayPtr)[Fp_string + "_old"];
  eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];
}

//
// J2 nonlinear system
//
template<typename EvalT>
class J2NLS:
    public Intrepid2::Function_Base<J2NLS<EvalT>, typename EvalT::ScalarT>
{
  using S = typename EvalT::ScalarT;

public:
  J2NLS(
      RealType sat_mod_,
      RealType sat_exp_,
      RealType eqps_old_,
      S const & K,
      S const & smag,
      S const & mubar,
      S const & Y) :
      sat_mod(sat_mod_),
      sat_exp(sat_exp_),
      eqps_old(eqps_old_),
      K_(K),
      smag_(smag),
      mubar_(mubar),
      Y_(Y)
  {
  }

  static constexpr Intrepid2::Index
  DIMENSION{1};

  static constexpr
  char const * const
  NAME{"J2 NLS"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<J2NLS<EvalT>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x)
  {
    // Firewalls.
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const
    K = peel<EvalT, T, N>()(K_);

    T const
    smag = peel<EvalT, T, N>()(smag_);

    T const
    mubar = peel<EvalT, T, N>()(mubar_);

    T const
    Y = peel<EvalT, T, N>()(Y_);

    // This is the actual computation of the gradient.
    Intrepid2::Vector<T, N>
    r(dimension);

    T const &
    X = x(0);

    T const
    alpha = eqps_old + sq23 * X;

    T const
    H = K * alpha + sat_mod * (1.0 - std::exp(-sat_exp * alpha));

    T const
    R = smag - (2.0 * mubar * X + sq23 * (Y + H));

    r(0) = R;

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<J2NLS<EvalT>, S>::hessian(*this, x);
  }

  // Constants.
  RealType const
  sq23{std::sqrt(2.0 / 3.0)};

  // RealType data (fixed non-AD type)
  RealType const
  sat_mod{0.0};

  RealType const
  sat_exp{0.0};

  RealType const
  eqps_old{0.0};

  // Inputs
  S const &
  K_;

  S const &
  smag_;

  S const &
  mubar_;

  S const &
  Y_;
};

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
J2MiniKernel<EvalT, Traits>::
operator()(int cell, int pt) const
{
  constexpr
  Intrepid2::Index
  MAX_DIM{3};

  Intrepid2::Tensor<ScalarT, MAX_DIM>
  F(num_dims_);

  Intrepid2::Tensor<ScalarT, MAX_DIM> const
  I(Intrepid2::eye<ScalarT, MAX_DIM>(num_dims_));

  Intrepid2::Tensor<ScalarT, MAX_DIM>
  sigma(num_dims_);
  
  ScalarT const
  kappa = elastic_modulus(cell, pt)
      / (3.0 * (1.0 - 2.0 * poissons_ratio(cell, pt)));

  ScalarT const
  mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));

  ScalarT const
  K = hardeningModulus(cell, pt);

  ScalarT const
  Y = yieldStrength(cell, pt);

  ScalarT const
  Jm23 = std::pow(J(cell, pt), -2. / 3.);

  // fill local tensors
  F.fill(def_grad, cell, pt, 0, 0);

  //Fpn.fill( &Fpold(cell,pt,int(0),int(0)) );

  Intrepid2::Tensor<ScalarT, MAX_DIM>
  Fpn(num_dims_);

  for (int i{0}; i < num_dims_; ++i) {
    for (int j{0}; j < num_dims_; ++j) {
      Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
    }
  }

  // compute trial state
  Intrepid2::Tensor<ScalarT, MAX_DIM> const
  Fpinv = Intrepid2::inverse(Fpn);

  Intrepid2::Tensor<ScalarT, MAX_DIM> const
  Cpinv = Fpinv * Intrepid2::transpose(Fpinv);

  Intrepid2::Tensor<ScalarT, MAX_DIM> const
  be = Jm23 * F * Cpinv * Intrepid2::transpose(F);

  Intrepid2::Tensor<ScalarT, MAX_DIM>
  s = mu * Intrepid2::dev(be);

  ScalarT const
  mubar = Intrepid2::trace(be) * mu / (num_dims_);

  // check yield condition
  ScalarT const
  smag = Intrepid2::norm(s);

  ScalarT const
  sq23{std::sqrt(2.0 / 3.0)};

  ScalarT const
  f = smag - sq23 * (Y + K * eqpsold(cell, pt)
      + sat_mod * (1.0 - std::exp(-sat_exp * eqpsold(cell, pt))));

  RealType constexpr
  yield_tolerance = 1.0e-12;

  if (f > yield_tolerance) {
    // Use minimization equivalent to return mapping
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using NLS = J2NLS<EvalT>;

    constexpr
    Intrepid2::Index
    nls_dim{NLS::DIMENSION};

    using MIN = Intrepid2::Minimizer<ValueT, nls_dim>;
    using STEP = Intrepid2::NewtonStep<NLS, ValueT, nls_dim>;

    MIN
    minimizer;

    STEP
    step;

    NLS
    j2nls(sat_mod, sat_exp, eqpsold(cell, pt), K, smag, mubar, Y);

    Intrepid2::Vector<ScalarT, nls_dim>
    x;

    x(0) = 0.0;

    LCM::MiniSolver<MIN, STEP, NLS, EvalT, nls_dim>
    mini_solver(minimizer, step, j2nls, x);

    ScalarT const
    alpha = eqpsold(cell, pt) + sq23 * x(0);

    ScalarT const
    H = K * alpha + sat_mod * (1.0 - exp(-sat_exp * alpha));

    ScalarT const
    dgam = x(0);

    // plastic direction
    Intrepid2::Tensor<ScalarT, MAX_DIM> const
    N = (1 / smag) * s;

    // update s
    s -= 2 * mubar * dgam * N;

    // update eqps
    eqps(cell, pt) = alpha;

    // mechanical source
    if (have_temperature_ == true && delta_time(0) > 0) {
      source(cell, pt) = (sq23 * dgam / delta_time(0)
          * (Y + H + temperature_(cell, pt))) / (density_ * heat_capacity_);
    }

    // exponential map to get Fpnew
    Intrepid2::Tensor<ScalarT, MAX_DIM> const
    A = dgam * N;

    Intrepid2::Tensor<ScalarT, MAX_DIM> const
    expA = Intrepid2::exp(A);

    Intrepid2::Tensor<ScalarT, MAX_DIM> const
    Fpnew = expA * Fpn;

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
  yieldSurf(cell, pt) = Y + K * eqps(cell, pt)
      + sat_mod * (1. - std::exp(-sat_exp * eqps(cell, pt)));

  // compute pressure
  ScalarT const
  p = 0.5 * kappa * (J(cell, pt) - 1. / (J(cell, pt)));

  // compute stress
  sigma = p * I + s / J(cell, pt);

  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      stress(cell, pt, i, j) = sigma(i, j);
    }
  }
  
  if (have_temperature_ == true) {
    F.fill(def_grad, cell, pt, 0, 0);

    ScalarT const
    J = Intrepid2::det(F);

    sigma.fill(stress, cell, pt, 0, 0);
    sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J * J))
        * (temperature_(cell, pt) - ref_temperature_) * I;
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        stress(cell, pt, i, j) = sigma(i, j);
      }
    }
  }
  
}
} // namespace LCM
