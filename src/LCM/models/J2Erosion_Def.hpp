//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ACEcommon.hpp"
#include "Albany_STKDiscretization.hpp"
#include "J2Erosion.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
J2ErosionKernel<EvalT, Traits>::J2ErosionKernel(
    ConstitutiveModel<EvalT, Traits>&    model,
    Teuchos::ParameterList*              p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model)
{
  this->setIntegrationPointLocationFlag(true);

  // Baseline constants
  sat_mod_         = p->get<RealType>("Saturation Modulus", 0.0);
  sat_exp_         = p->get<RealType>("Saturation Exponent", 0.0);
  erosion_rate_    = p->get<RealType>("Erosion Rate", 0.0);
  element_size_    = p->get<RealType>("Element Size", 0.0);
  critical_stress_ = p->get<RealType>("Critical Stress", 0.0);
  critical_angle_  = p->get<RealType>("Critical Angle", 0.0);

  if (p->isParameter("Time File") == true) {
    std::string const filename = p->get<std::string>("Time File");
    time_                      = vectorFromFile(filename);
  }

  if (p->isParameter("Sea Level File") == true) {
    std::string const filename = p->get<std::string>("Sea Level File");
    sea_level_                 = vectorFromFile(filename);
  }

  ALBANY_ASSERT(
      time_.size() == sea_level_.size(),
      "*** ERROR: Number of times and number of sea level values must match");

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
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);
  setDependentField("Yield Strength", dl->qp_scalar);
  setDependentField("Hardening Modulus", dl->qp_scalar);
  setDependentField("Delta Time", dl->workset_scalar);

  // define the evaluated fields
  setEvaluatedField("Failure Indicator", dl->cell_scalar);
  setEvaluatedField("Exposure Time", dl->qp_scalar);
  setEvaluatedField(cauchy_string, dl->qp_tensor);
  setEvaluatedField(Fp_string, dl->qp_tensor);
  setEvaluatedField(eqps_string, dl->qp_scalar);
  setEvaluatedField(yieldSurface_string, dl->qp_scalar);
  if (have_temperature_ == true) {
    setDependentField("Temperature", dl->qp_scalar);
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
  //
  // mechanical source
  if (have_temperature_ == true) {
    addStateVariable(
        "Temperature",
        dl->qp_scalar,
        "scalar",
        0.0,
        true,
        p->get<bool>("Output Temperature", false));

    addStateVariable(
        source_string,
        dl->qp_scalar,
        "scalar",
        0.0,
        false,
        p->get<bool>("Output Mechanical Source", false));
  }

  // failed state
  addStateVariable(
      "Failure Indicator",
      dl->cell_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Failure Indicator", true));

  // exposure time
  addStateVariable(
      "Exposure Time",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Exposure Time", true));
}

template <typename EvalT, typename Traits>
void
J2ErosionKernel<EvalT, Traits>::init(
    Workset&                 workset,
    FieldMap<const ScalarT>& dep_fields,
    FieldMap<ScalarT>&       eval_fields)
{
  std::string cauchy_string       = field_name_map_["Cauchy_Stress"];
  std::string Fp_string           = field_name_map_["Fp"];
  std::string eqps_string         = field_name_map_["eqps"];
  std::string yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string source_string       = field_name_map_["Mechanical_Source"];
  std::string F_string            = field_name_map_["F"];
  std::string J_string            = field_name_map_["J"];

  // extract dependent MDFields
  def_grad_          = *dep_fields[F_string];
  J_                 = *dep_fields[J_string];
  poissons_ratio_    = *dep_fields["Poissons Ratio"];
  elastic_modulus_   = *dep_fields["Elastic Modulus"];
  yield_strength_    = *dep_fields["Yield Strength"];
  hardening_modulus_ = *dep_fields["Hardening Modulus"];
  delta_time_        = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  stress_        = *eval_fields[cauchy_string];
  Fp_            = *eval_fields[Fp_string];
  eqps_          = *eval_fields[eqps_string];
  yield_surf_    = *eval_fields[yieldSurface_string];
  failed_        = *eval_fields["Failure Indicator"];
  exposure_time_ = *eval_fields["Exposure Time"];

  if (have_temperature_ == true) {
    source_      = *eval_fields[source_string];
    temperature_ = *dep_fields["Temperature"];
  }

  // get State Variables
  Fp_old_   = (*workset.stateArrayPtr)[Fp_string + "_old"];
  eqps_old_ = (*workset.stateArrayPtr)[eqps_string + "_old"];

  auto& disc               = *workset.disc;
  auto& stk_disc           = dynamic_cast<Albany::STKDiscretization&>(disc);
  auto& mesh_struct        = *(stk_disc.getSTKMeshStruct());
  auto& field_cont         = *(mesh_struct.getFieldContainer());
  have_boundary_indicator_ = field_cont.hasBoundaryIndicatorField();

  elemWsLIDGIDMap_ = stk_disc.getElemWsLIDGIDMap();
  ws_index_ = workset.wsIndex;

  if (have_boundary_indicator_ == true) {
    boundary_indicator_ = workset.boundary_indicator;
    ALBANY_ASSERT(boundary_indicator_.is_null() == false);
  }

  current_time_ = workset.current_time;

  auto const num_cells = workset.numCells;
  for (auto cell = 0; cell < num_cells; ++cell) { failed_(cell, 0) = 0.0; }
}

//
// J2 nonlinear system
//
template <typename EvalT, minitensor::Index M = 1>
class J2ErosionNLS
    : public minitensor::
          Function_Base<J2ErosionNLS<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

 public:
  J2ErosionNLS(
      RealType sat_mod,
      RealType sat_exp,
      RealType eqps_old,
      S const& K,
      S const& smag,
      S const& mubar,
      S const& Y)
      : sat_mod_(sat_mod),
        sat_exp_(sat_exp),
        eqps_old_(eqps_old),
        K_(K),
        smag_(smag),
        mubar_(mubar),
        Y_(Y)
  {
  }

  constexpr static char const* const NAME{"J2 NLS"};

  using Base = minitensor::
      Function_Base<J2ErosionNLS<EvalT, M>, typename EvalT::ScalarT, M>;

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
    T const  alpha = eqps_old_ + SQ23 * X;
    T const  H     = K * alpha + sat_mod_ * (1.0 - std::exp(-sat_exp_ * alpha));
    T const  R     = smag - (2.0 * mubar * X + SQ23 * (Y + H));

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
  RealType const sat_mod_{0.0};
  RealType const sat_exp_{0.0};
  RealType const eqps_old_{0.0};

  // Inputs
  S const& K_;
  S const& smag_;
  S const& mubar_;
  S const& Y_;
};

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
J2ErosionKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  constexpr minitensor::Index MAX_DIM{3};
  using Tensor = minitensor::Tensor<ScalarT, MAX_DIM>;
  Tensor       F(num_dims_);
  Tensor const I(minitensor::eye<ScalarT, MAX_DIM>(num_dims_));
  Tensor       sigma(num_dims_);

  auto const coords       = this->model_.getCoordVecField();
  auto const height       = Sacado::Value<ScalarT>::eval(coords(cell, pt, 2));
  auto const current_time = current_time_;

  ScalarT const E     = elastic_modulus_(cell, pt);
  ScalarT const nu    = poissons_ratio_(cell, pt);
  ScalarT const kappa = E / (3.0 * (1.0 - 2.0 * nu));
  ScalarT const mu    = E / (2.0 * (1.0 + nu));
  ScalarT const K     = hardening_modulus_(cell, pt);
  ScalarT const Y     = yield_strength_(cell, pt);
  ScalarT const J1    = J_(cell, pt);
  ScalarT const Jm23  = 1.0 / std::cbrt(J1 * J1);

  auto&& delta_time    = delta_time_(0);
  auto&& failed        = failed_(cell, 0);
  auto&& exposure_time = exposure_time_(cell, pt);

  auto const proc_rank = Albany::getProcRank();
  if (pt == 0) {
    auto ws_lid = std::make_pair(ws_index_, cell);
    auto iter = elemWsLIDGIDMap_.find(ws_lid);
    ALBANY_ASSERT(iter != elemWsLIDGIDMap_.end());
    auto gid = iter->second;
    std::cout << "**** DEBUG MATE PROC GID: " << proc_rank << " "
              << std::setw(3) << std::setfill('0') << gid << "\n";
  }

  // Determine if erosion has occurred.
  auto const erosion_rate = erosion_rate_;
  auto const element_size = element_size_;
  bool const is_erodible  = erosion_rate > 0.0;
  auto const critical_exposure_time =
      is_erodible == true ? element_size / erosion_rate : 0.0;

  auto const sea_level =
      sea_level_.size() > 0 ?
          interpolateVectors(time_, sea_level_, current_time) :
          0.0;
  bool const is_exposed_to_water = (height <= sea_level);
  bool const is_at_boundary =
      have_boundary_indicator_ == true ?
          static_cast<bool>(*(boundary_indicator_[cell])) :
          false;

  bool const is_erodible_at_boundary = is_erodible && is_at_boundary;
  if (is_erodible_at_boundary == true) {
    if (is_exposed_to_water == true) { exposure_time += delta_time; }
    if (exposure_time >= critical_exposure_time) {
      failed += 1.0;
      exposure_time = 0.0;
    }
  }

  // fill local tensors
  F.fill(def_grad_, cell, pt, 0, 0);

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
      Fpn(i, j) = ScalarT(Fp_old_(cell, pt, i, j));
    }
  }

  // compute trial state
  Tensor const  Fpinv = minitensor::inverse(Fpn);
  Tensor const  Cpinv = Fpinv * minitensor::transpose(Fpinv);
  Tensor const  be    = Jm23 * Fm * Cpinv * minitensor::transpose(Fm);
  Tensor        s     = mu * minitensor::dev(be);
  ScalarT const mubar = minitensor::trace(be) * mu / (num_dims_);

  // check yield condition
  ScalarT const smag = minitensor::norm(s);
  ScalarT const f =
      smag -
      SQ23 * (Y + K * eqps_old_(cell, pt) +
              sat_mod_ * (1.0 - std::exp(-sat_exp_ * eqps_old_(cell, pt))));

  RealType constexpr yield_tolerance = 1.0e-12;

  if (f > yield_tolerance) {
    // Use minimization equivalent to return mapping
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using NLS    = J2ErosionNLS<EvalT>;

    constexpr minitensor::Index nls_dim{NLS::DIMENSION};

    using MIN  = minitensor::Minimizer<ValueT, nls_dim>;
    using STEP = minitensor::NewtonStep<NLS, ValueT, nls_dim>;

    MIN  minimizer;
    STEP step;
    NLS  j2nls(sat_mod_, sat_exp_, eqps_old_(cell, pt), K, smag, mubar, Y);

    minitensor::Vector<ScalarT, nls_dim> x;

    x(0) = 0.0;

    LCM::MiniSolver<MIN, STEP, NLS, EvalT, nls_dim> mini_solver(
        minimizer, step, j2nls, x);

    ScalarT const alpha = eqps_old_(cell, pt) + SQ23 * x(0);
    ScalarT const H     = K * alpha + sat_mod_ * (1.0 - exp(-sat_exp_ * alpha));
    ScalarT const dgam  = x(0);

    // plastic direction
    Tensor const N = (1 / smag) * s;

    // update s
    s -= 2 * mubar * dgam * N;

    // update eqps
    eqps_(cell, pt) = alpha;

    // mechanical source
    if (have_temperature_ == true && delta_time_(0) > 0) {
      source_(cell, pt) =
          (SQ23 * dgam / delta_time_(0) * (Y + H + temperature_(cell, pt))) /
          (density_ * heat_capacity_);
    }

    // exponential map to get Fpnew
    Tensor const A     = dgam * N;
    Tensor const expA  = minitensor::exp(A);
    Tensor const Fpnew = expA * Fpn;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) { Fp_(cell, pt, i, j) = Fpnew(i, j); }
    }
  } else {
    eqps_(cell, pt) = eqps_old_(cell, pt);

    if (have_temperature_ == true) source_(cell, pt) = 0.0;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) { Fp_(cell, pt, i, j) = Fpn(i, j); }
    }
  }

  // update yield surface
  yield_surf_(cell, pt) =
      Y + K * eqps_(cell, pt) +
      sat_mod_ * (1. - std::exp(-sat_exp_ * eqps_(cell, pt)));

  // compute pressure
  ScalarT const p = 0.5 * kappa * (J_(cell, pt) - 1. / (J_(cell, pt)));

  // compute stress
  sigma = p * I + s / J_(cell, pt);

  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      stress_(cell, pt, i, j) = sigma(i, j);
    }
  }

  //
  // Determine if critical stress is exceeded
  //

  // sigma_XX component for now
  auto const critical_stress = critical_stress_;
  if (critical_stress > 0.0) {
    auto const stress_test = Sacado::Value<ScalarT>::eval(sigma(0, 0));
    if (std::abs(stress_test) >= critical_stress) failed += 1.0;
  }

  //
  // Determine if kinematic failure occurred
  //
  auto const critical_angle = critical_angle_;
  if (critical_angle > 0.0) {
    auto const Fval   = Sacado::Value<decltype(F)>::eval(F);
    auto const Q      = minitensor::polar_rotation(Fval);
    auto       cosine = 0.5 * (minitensor::trace(Q) - 1.0);
    cosine            = cosine > 1.0 ? 1.0 : cosine;
    cosine            = cosine < -1.0 ? -1.0 : cosine;
    auto const theta  = std::acos(cosine);
    if (std::abs(theta) >= critical_angle) failed += 1.0;
  }
}
}  // namespace LCM
