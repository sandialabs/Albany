//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(CrystalPlasticityCore_hpp)
#define CrystalPlasticityCore_hpp

#include <MiniNonlinearSolver.h>

namespace CP
{

static constexpr Intrepid2::Index 
MAX_DIM = 3;

static constexpr Intrepid2::Index
MAX_SLIP = 12;

static constexpr Intrepid2::Index
NLS_DIM = 2 * MAX_SLIP;

static constexpr Intrepid2::Index
MAX_FAMILY = 3;

enum class FlowRuleType
{
  UNDEFINED = 0,
  POWER_LAW = 1,
  THERMAL_ACTIVATION = 2,
  POWER_LAW_DRAG = 3
};

enum class HardeningLawType
{
  UNDEFINED = 0, 
  LINEAR_MINUS_RECOVERY = 1, 
  SATURATION = 2, 
  DISLOCATION_DENSITY = 3
};



//
//! Struct containing slip system information.
//
template<Intrepid2::Index NumDimT>
struct SlipSystem
{

  SlipSystem() {}

  // SlipSystem(SlipFamily<MAX_SLIP> const & sf)
  // : slip_family_{sf} {}

  Intrepid2::Index
  slip_family_index_;

  //! Slip system vectors.
  Intrepid2::Vector<RealType, NumDimT>
  s_;

  Intrepid2::Vector<RealType, NumDimT>
  n_;

  //! Schmid Tensor.
  Intrepid2::Tensor<RealType, NumDimT> 
  projector_;

  //
  RealType
  state_hardening_initial_;

  // SlipFamily<MAX_SLIP> const &
  // slip_family_;
};

// Forward declarations
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct HardeningParameterBase;

struct FlowParameterBase;


//
// Slip system family - collection of slip systems grouped by flow and
// hardening characteristics
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct SlipFamily
{
  SlipFamily() {}

  ~SlipFamily() {}

  Intrepid2::Index
  num_slip_sys_{0};

  Intrepid2::Vector<Intrepid2::Index, NumSlipT>
  slip_system_indices_;

  std::unique_ptr<HardeningParameterBase<NumDimT, NumSlipT>>
  phardening_parameters_{nullptr};

  std::unique_ptr<FlowParameterBase>
  pflow_parameters_{nullptr};

  Intrepid2::Tensor<RealType, NumSlipT>
  latent_matrix_;

  HardeningLawType
  type_hardening_law_{HardeningLawType::UNDEFINED};

  FlowRuleType
  type_flow_rule_{FlowRuleType::UNDEFINED};
};

///
/// Hardening Base
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct HardeningParameterBase
{
  using ParamIndex = int;
  
  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems);

  void
  setParameter(ParamIndex const index_param, RealType const value_param)
  {
    hardening_params_[index_param] = value_param;
  }

  RealType
  getParameter(ParamIndex const index_param)
  {
    return hardening_params_[index_param];
  }

  virtual
  ~HardeningParameterBase() {}

  std::map<std::string, ParamIndex>
  param_map_;

  Intrepid2::Vector<RealType>
  hardening_params_;
};


///
/// Linear hardening with recovery
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct LinearMinusRecoveryHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  using ParamIndex = typename HardeningParameterBase<NumDimT, NumSlipT>::ParamIndex;

  enum HardeningParamTypes : ParamIndex
  {
    RESISTANCE_SLIP_INITIAL,
    MODULUS_HARDENING,
    MODULUS_RECOVERY,
    NUM_PARAMS
  };

  LinearMinusRecoveryHardeningParameters()
  {
    this->param_map_["Tau Critical"] = RESISTANCE_SLIP_INITIAL;
    this->param_map_["Initial Slip Resistance"] = RESISTANCE_SLIP_INITIAL;
    this->param_map_["Hardening"] = MODULUS_HARDENING;
    this->param_map_["Hardening Modulus"] = MODULUS_HARDENING;
    this->param_map_["Hardening Exponent"] = MODULUS_RECOVERY;
    this->param_map_["Recovery Modulus"] = MODULUS_RECOVERY;
    this->param_map_["Initial Hardening State"] = RESISTANCE_SLIP_INITIAL;
    this->hardening_params_.set_dimension(NUM_PARAMS);
    this->hardening_params_.fill(Intrepid2::ZEROS);
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems);

  virtual
  ~LinearMinusRecoveryHardeningParameters() {}
};

///
/// Saturation hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct SaturationHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  using ParamIndex = typename HardeningParameterBase<NumDimT, NumSlipT>::ParamIndex;

  enum HardeningParamTypes : ParamIndex
  {
    RESISTANCE_SLIP_INITIAL,
    RATE_HARDENING,
    STRESS_SATURATION_INITIAL,
    EXPONENT_SATURATION,
    RATE_SLIP_REFERENCE,
    NUM_PARAMS
  };

  SaturationHardeningParameters()
  {
    this->param_map_["Initial Slip Resistance"] = RESISTANCE_SLIP_INITIAL;
    this->param_map_["Hardening Rate"] = RATE_HARDENING;
    this->param_map_["Initial Saturation Stress"] = STRESS_SATURATION_INITIAL;
    this->param_map_["Saturation Exponent"] = EXPONENT_SATURATION;
    this->param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    this->param_map_["Initial Hardening State"] = RESISTANCE_SLIP_INITIAL;
    this->hardening_params_.set_dimension(NUM_PARAMS);
    this->hardening_params_.fill(Intrepid2::ZEROS);
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems);

  virtual
  ~SaturationHardeningParameters() {}
};

///
/// Dislocation-density based hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct DislocationDensityHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  using ParamIndex = typename HardeningParameterBase<NumDimT, NumSlipT>::ParamIndex;

  enum HardeningParamTypes : ParamIndex
  {
    FACTOR_GEOMETRY_DISLOCATION,
    DENSITY_DISLOCATION_INITIAL,
    FACTOR_GENERATION,
    FACTOR_ANNIHILATION,
    MODULUS_SHEAR,
    MAGNITUDE_BURGERS,
    NUM_PARAMS
  };

  DislocationDensityHardeningParameters()
  {
    this->param_map_["Geometric Factor"] = FACTOR_GEOMETRY_DISLOCATION;
    this->param_map_["Initial Dislocation Density"] = DENSITY_DISLOCATION_INITIAL;
    this->param_map_["Generation Factor"] = FACTOR_GENERATION;
    this->param_map_["Annihilation Factor"] = FACTOR_ANNIHILATION;
    this->param_map_["Shear Modulus"] = MODULUS_SHEAR;
    this->param_map_["Burgers Vector Magnitude"] = MAGNITUDE_BURGERS;
    this->param_map_["Initial Hardening State"] = DENSITY_DISLOCATION_INITIAL;
    this->hardening_params_.set_dimension(NUM_PARAMS);
    this->hardening_params_.fill(Intrepid2::ZEROS);
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems);

  virtual
  ~DislocationDensityHardeningParameters() {}
};

///
/// No hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct NoHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  NoHardeningParameters()
  {
    return;
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems);

  virtual
  ~NoHardeningParameters() {}
};

//
// Factory returning a pointer to a hardening paremeter object
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
std::unique_ptr<HardeningParameterBase<NumDimT, NumSlipT>>
hardeningParameterFactory(HardeningLawType type_hardening_law)
{
  using HPUP = std::unique_ptr<HardeningParameterBase<NumDimT, NumSlipT>>;

  switch (type_hardening_law) {

  default:
    std::cerr << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "ERROR: Unknown hardening law\n";
    exit(1);
    break;

  case HardeningLawType::LINEAR_MINUS_RECOVERY:
    return HPUP(new LinearMinusRecoveryHardeningParameters<NumDimT, NumSlipT>());
    break;

  case HardeningLawType::SATURATION:
    return HPUP(new SaturationHardeningParameters<NumDimT, NumSlipT>());
    break;

  case HardeningLawType::DISLOCATION_DENSITY:
    return HPUP(new DislocationDensityHardeningParameters<NumDimT, NumSlipT>());
    break;

  case HardeningLawType::UNDEFINED:
    return HPUP(new NoHardeningParameters<NumDimT, NumSlipT>());
    break;

  }

  return HPUP(nullptr);
}


///
/// FlowRule parameters base class
///
struct FlowParameterBase
{
  using ParamIndex = int;

  void
  setParameter(ParamIndex const index_param, RealType const value_param)
  {
    flow_params_[index_param] = value_param;
  }

  RealType
  getParameter(ParamIndex const index_param)
  {
    return flow_params_[index_param];
  }

  std::map<std::string, ParamIndex>
  param_map_;

  // Flow parameters
  Intrepid2::Vector<RealType>
  flow_params_;
};

///
/// Power Law parameters
///
struct PowerLawFlowParameters final : public FlowParameterBase
{
  enum FlowParamTypes : FlowParameterBase::ParamIndex
  {
    RATE_SLIP_REFERENCE,
    EXPONENT_RATE,
    NUM_PARAMS
  };

  PowerLawFlowParameters()
  {
    param_map_["Gamma Dot"] = RATE_SLIP_REFERENCE;
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Gamma Exponent"] = EXPONENT_RATE;
    param_map_["Rate Exponent"] = EXPONENT_RATE;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(Intrepid2::ZEROS);
  }
};


///
/// Thermal activation parameters
///
struct ThermalActivationFlowParameters final : public FlowParameterBase
{
  enum FlowParamTypes : FlowParameterBase::ParamIndex
  {
    RATE_SLIP_REFERENCE,
    ENERGY_ACTIVATION,
    NUM_PARAMS
  };

  ThermalActivationFlowParameters()
  {
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Activation Energy"] = ENERGY_ACTIVATION;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(Intrepid2::ZEROS);
  }
};


///
/// Power Law with Viscous Drag parameters
///
struct PowerLawDragFlowParameters final : public FlowParameterBase
{
  enum FlowParamTypes : FlowParameterBase::ParamIndex
  {
    RATE_SLIP_REFERENCE,
    EXPONENT_RATE,
    DRAG_COEFF,
    NUM_PARAMS
  };

  PowerLawDragFlowParameters()
  {
    param_map_["Gamma Dot"] = RATE_SLIP_REFERENCE;
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Gamma Exponent"] = EXPONENT_RATE;
    param_map_["Rate Exponent"] = EXPONENT_RATE;
    param_map_["Drag Coefficient"] = DRAG_COEFF;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(Intrepid2::ZEROS);
  }
};


///
/// No flow (elasticity) parameters
///
struct NoFlowParameters final : public FlowParameterBase
{
  NoFlowParameters()
  {
    return;
  }
};


//
// Factory returning a pointer to a Flow parameters object
//
std::unique_ptr<FlowParameterBase>
flowParameterFactory(FlowRuleType type_flow_rule)
{
  using FPUP = std::unique_ptr<FlowParameterBase>;

  switch (type_flow_rule) {

  default:
    std::cerr << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "ERROR: Unknown flow rule\n";
    exit(1);
    break;

  case FlowRuleType::POWER_LAW:
    return FPUP(new PowerLawFlowParameters());
    break;

  case FlowRuleType::POWER_LAW_DRAG:
    return FPUP(new PowerLawDragFlowParameters());
    break;

  case FlowRuleType::THERMAL_ACTIVATION:
    return FPUP(new ThermalActivationFlowParameters());
    break;

  case FlowRuleType::UNDEFINED:
    return FPUP(new NoFlowParameters());
    break;

  }

  return FPUP(nullptr);
}

//
//! Check tensor for NaN and inf values.
//
template<Intrepid2::Index NumDimT, typename ArgT>
void
confirmTensorSanity(
    Intrepid2::Tensor<ArgT, NumDimT> const & input,
    std::string const & message);


//
//! Compute Lp_np1 and Fp_np1 based on computed slip increment.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
applySlipIncrement(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
    Intrepid2::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid2::Tensor<ArgT, NumDimT> & Fp_np1);



//
//! Update the hardness.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
updateHardness(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);



///
/// Update the plastic slips
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
updateSlip(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_resistance,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1);



//
//! Compute stress.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT, typename DataT>
void
computeStress(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    Intrepid2::Tensor4<DataT, NumDimT> const & C,
    Intrepid2::Tensor<DataT, NumDimT> const & F,
    Intrepid2::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear);



//
//! Construct elasticity tensor
//
template<Intrepid2::Index NumDimT, typename DataT, typename ArgT>
void
computeCubicElasticityTensor(
    DataT c11, 
    DataT c12, 
    DataT c44,
    Intrepid2::Tensor4<ArgT, NumDimT> & C);



//
//! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip 
//  increments as unknowns.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
class ResidualSlipNLS:
    public Intrepid2::Function_Base<
    ResidualSlipNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using ScalarT = typename EvalT::ScalarT;

public:

  //! Constructor.
  ResidualSlipNLS(
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<SlipSystem<NumDimT>> const & slip_systems,
      std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

  static constexpr char const * const NAME =
      "Crystal Plasticity Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the slip 
  // at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x);


  //! Default implementation of hessian.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

private:

  RealType
  num_dim_;

  RealType
  num_slip_;

  Intrepid2::Tensor4<ScalarT, NumDimT> const &
  C_;

  std::vector<SlipSystem<NumDimT>> const &
  slip_systems_;

  std::vector<SlipFamily<NumDimT, NumSlipT>> const &
  slip_families_;

  Intrepid2::Tensor<RealType, NumDimT> const &
  Fp_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  state_hardening_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  slip_n_;

  Intrepid2::Tensor<ScalarT, NumDimT> const &
  F_np1_;

  RealType
  dt_;
};



//
//! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip 
//  increments and hardnesses as unknowns.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
class ResidualSlipHardnessNLS:
    public Intrepid2::Function_Base<
    ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using ScalarT = typename EvalT::ScalarT;

public:

  //! Constructor.
  ResidualSlipHardnessNLS(
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<SlipSystem<NumDimT>> const & slip_systems,
      std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

  static constexpr char const * const NAME =
      "Slip and Hardness Residual Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the slip 
  // and hardness at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x);


  //! Default implementation of hessian.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

private:

  RealType
  num_dim_;

  RealType
  num_slip_;

  Intrepid2::Tensor4<ScalarT, NumDimT> const &
  C_;

  std::vector<SlipSystem<NumDimT>> const &
  slip_systems_;

  std::vector<SlipFamily<NumDimT, NumSlipT>> const &
  slip_families_;

  Intrepid2::Tensor<RealType, NumDimT> const &
  Fp_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  state_hardening_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  slip_n_;

  Intrepid2::Tensor<ScalarT, NumDimT> const &
  F_np1_;

  RealType
  dt_;
};


//
// Nonlinear Solver (NLS) class for the CrystalPlasticity model explicit update
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
class ExplicitUpdateNLS:
    public Intrepid2::Function_Base<
    ExplicitUpdateNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using ScalarT = typename EvalT::ScalarT;

public:

  //! Constructor.
  ExplicitUpdateNLS(
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      std::vector<SlipSystem<NumDimT>> const & slip_systems,
      std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

  static constexpr char const * const NAME =
      "Slip and Hardness Residual Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the slip 
  // and hardness at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x);


  //! Default implementation of hessian.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

private:

  RealType
  num_dim_;

  RealType
  num_slip_;

  Intrepid2::Tensor4<ScalarT, NumDimT> const &
  C_;

  std::vector<SlipSystem<NumDimT>> const &
  slip_systems_;

  std::vector<SlipFamily<NumDimT, NumSlipT>> const &
  slip_families_;

  Intrepid2::Tensor<RealType, NumDimT> const &
  Fp_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  state_hardening_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  slip_n_;

  Intrepid2::Tensor<ScalarT, NumDimT> const &
  F_np1_;

  RealType
  dt_;
};


///
/// FlowRule base class
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct FlowRuleBase
{
  FlowRuleBase() {}

  virtual
  ScalarT
  computeRateSlip(
    std::unique_ptr<FlowParameterBase> pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~FlowRuleBase() {}
};

///
/// Power Law
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct PowerLawFlowRule final : public FlowRuleBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::unique_ptr<FlowParameterBase> pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~PowerLawFlowRule() {}
};

///
/// Thermally-activated flow rule
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct ThermalActivationFlowRule final : public FlowRuleBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::unique_ptr<FlowParameterBase> pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~ThermalActivationFlowRule() {}
};


///
/// Power Law with Viscous Drag
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct PowerLawDragFlowRule final : public FlowRuleBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::unique_ptr<FlowParameterBase> pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~PowerLawDragFlowRule() {}
};


///
/// No flow (elasticity)
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct NoFlowRule final : public FlowRuleBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::unique_ptr<FlowParameterBase> pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~NoFlowRule() {}
};


//
// Factory returning a pointer to a flow rule object
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
class flowRuleFactory
{

public:

  using FlowRuleBaseType = FlowRuleBase<NumDimT, NumSlipT, ScalarT>;

  FlowRuleBaseType *
  createFlowRule(FlowRuleType type_flow_rule) {  

    switch (type_flow_rule) {

    default:
      std::cerr << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "ERROR: Unknown flow rule\n";
      exit(1);
      break;

    case FlowRuleType::POWER_LAW:
      return new(flow_buffer_) PowerLawFlowRule<NumDimT, NumSlipT, ScalarT>();
      break;

    case FlowRuleType::POWER_LAW_DRAG:
      return new(flow_buffer_) PowerLawDragFlowRule<NumDimT, NumSlipT, ScalarT>();
      break;

    case FlowRuleType::THERMAL_ACTIVATION:
      return new(flow_buffer_) ThermalActivationFlowRule<NumDimT, NumSlipT, ScalarT>();
      break;

    case FlowRuleType::UNDEFINED:
      return new(flow_buffer_) NoFlowRule<NumDimT, NumSlipT, ScalarT>();
      break;
    }

    return nullptr;
  }

private:

  unsigned char
  flow_buffer_[sizeof(FlowRuleBaseType)];
};

///
/// Hardening Law Base
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct HardeningLawBase
{
  HardeningLawBase() {}

  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance);

  virtual
  ~HardeningLawBase() {}
};

///
/// Linear hardening with recovery
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct LinearMinusRecoveryHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance);

  virtual
  ~LinearMinusRecoveryHardeningLaw() {}
};

///
/// Saturation hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct SaturationHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance);

  virtual
  ~SaturationHardeningLaw() {}
};

///
/// Dislocation-density based hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct DislocationDensityHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance);

  virtual
  ~DislocationDensityHardeningLaw() {}
};

///
/// No hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
struct NoHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ScalarT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ScalarT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ScalarT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ScalarT, NumSlipT> & slip_resistance);

  virtual
  ~NoHardeningLaw() {}
};

//
//
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ScalarT>
class hardeningLawFactory
{

public:

  using HardeningBaseType = HardeningLawBase<NumDimT, NumSlipT, ScalarT>;

  HardeningBaseType *
  createHardeningLaw(HardeningLawType type_hardening_law) {

    switch (type_hardening_law) {

      default:
        std::cerr << __PRETTY_FUNCTION__ << '\n';
        std::cerr << "ERROR: Unknown hardening law\n";
        exit(1);
        break;

      case HardeningLawType::LINEAR_MINUS_RECOVERY:
        return new(hardening_buffer_) LinearMinusRecoveryHardeningLaw<NumDimT, NumSlipT, ScalarT>();
        break;

      case HardeningLawType::SATURATION:
        return new(hardening_buffer_) SaturationHardeningLaw<NumDimT, NumSlipT, ScalarT>();
        break;

      case HardeningLawType::DISLOCATION_DENSITY:
        return new(hardening_buffer_) DislocationDensityHardeningLaw<NumDimT, NumSlipT, ScalarT>();
        break;

      case HardeningLawType::UNDEFINED:
        return new(hardening_buffer_) NoHardeningLaw<NumDimT, NumSlipT, ScalarT>();
        break;
    }

    return nullptr;
  }

private:

  unsigned char
  hardening_buffer_[sizeof(HardeningBaseType)];
  
};

} // namespace CP

#include "CrystalPlasticityCore_Def.hpp"

#endif
