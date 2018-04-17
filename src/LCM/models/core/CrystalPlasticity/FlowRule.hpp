//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "../../../../utility/StaticAllocator.hpp"

#if !defined(LCM_FlowRule_hpp)
#define LCM_FlowRule_hpp

namespace CP
{
/**
 *	Various types of flow rules that can be used.
 *
 *	FlowRuleType can be used to drive the factory function flowParameterFactory.
 */
enum class FlowRuleType
{
  UNDEFINED = 0,
  POWER_LAW = 1,
  THERMAL_ACTIVATION = 2,
  POWER_LAW_DRAG = 3
};


/**
 *	Factory returning a pointer to a Flow parameters object.
 *
 *	\param type_flow_rule	Which flow rule to instantiate.
 */
std::shared_ptr<FlowParameterBase>
flowParameterFactory(FlowRuleType type_flow_rule);


/**
 *	FlowRule parameters base class.
 *
 *	FlowRule parameters specify the various parameters used by a particular flow
 *	rule. All flow rule parameters derive from this class.
 */
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

  void
  setMaxIncrement(const RealType& max_incr)
  {
    max_incr_ = max_incr;
  }

  virtual
  void
  setTolerance() = 0;

  virtual
  ~FlowParameterBase()
  {
    return;
  }

  RealType
  min_tol_{TINY};

  RealType
  max_tol_{HUGE_};

  RealType
  max_incr_{LOG_HUGE};

  std::map<std::string, ParamIndex>
  param_map_;

  // Flow parameters
  minitensor::Vector<RealType>
  flow_params_;
};


/**
 *	Parameters for the Power Law flow rule.
 */
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
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Rate Exponent"] = EXPONENT_RATE;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(minitensor::Filler::ZEROS);
  }

  virtual
  void
  setTolerance()
  {
    min_tol_ =
      std::pow(2.0 * TINY, 0.5 / flow_params_(EXPONENT_RATE));

    max_tol_ =
      std::pow(0.5 * HUGE_, 0.5 / flow_params_(EXPONENT_RATE));
  }
};


/**
 *	Parameters for the Thermal Activation flow rule.
 */
struct ThermalActivationFlowParameters final : public FlowParameterBase
{
  enum FlowParamTypes : FlowParameterBase::ParamIndex
  {
    RATE_SLIP_REFERENCE,
    ENERGY_ACTIVATION,
    RESISTANCE_THERMAL,
    EXPONENT_P,
    EXPONENT_Q,
    NUM_PARAMS
  };

  ThermalActivationFlowParameters()
  {
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Activation Energy"] = ENERGY_ACTIVATION;
    param_map_["Thermal Resistance"] = RESISTANCE_THERMAL;
    param_map_["P Exponent"] = EXPONENT_P;
    param_map_["Q Exponent"] = EXPONENT_Q;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(minitensor::Filler::ZEROS);
  }

  virtual
  void
  setTolerance()
  {
    min_tol_ = 2.0 * TINY;
    max_tol_ = 0.5 * HUGE_;
  }
};


/**
 *	Parameters for the Power Law with Viscous Drag flow rule.
 */
struct PowerLawDragFlowParameters final : public FlowParameterBase
{
  enum FlowParamTypes : FlowParameterBase::ParamIndex
  {
    RATE_SLIP_REFERENCE,
    EXPONENT_RATE,
    COEFFICIENT_DRAG,
    NUM_PARAMS
  };

  PowerLawDragFlowParameters()
  {
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Rate Exponent"] = EXPONENT_RATE;
    param_map_["Drag Coefficient"] = COEFFICIENT_DRAG;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(minitensor::Filler::ZEROS);
  }

  virtual
  void
  setTolerance()
  {
    min_tol_ =
      std::pow(2.0 * TINY, 0.5 / flow_params_(EXPONENT_RATE));

    max_tol_ =
      std::pow(0.5 * HUGE_, 0.5 / flow_params_(EXPONENT_RATE));
  }
};


/**
 *	Parameters for the no flow (elasticity) rule.
 */
struct NoFlowParameters final : public FlowParameterBase
{
  NoFlowParameters()
  {
    return;
  }

  virtual
  void
  setTolerance()
  {
    return;
  }
};


/**
 *  Base class for flow rules.
 *
 *  \tparam ArgT The scalar type for computing the slip rate.
 */
template<typename ArgT>
struct FlowRuleBase
{
  FlowRuleBase() {}

  virtual
  ArgT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ArgT const & shear,
    ArgT const & slip_resistance,
    bool & failed) = 0;

  virtual
  ~FlowRuleBase() {}
};


/**
 *  Factory class for instantiating flow rules.
 */
class FlowRuleFactory
{

public:

  FlowRuleFactory() {};

  template<typename ArgT>
  utility::StaticPointer<FlowRuleBase<ArgT>>
  createFlowRule(FlowRuleType type_flow_rule) const;

private:

  mutable utility::StaticStackAllocator<sizeof(std::uintptr_t)> allocator_;
};


/**
 *  Power Law flow rule.
 *
 *  \tparam ArgT The scalar type for computing the slip rate.
 */
template<typename ArgT>
struct PowerLawFlowRule final : public FlowRuleBase<ArgT>
{
  virtual
  ArgT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ArgT const & shear,
    ArgT const & slip_resistance,
    bool & failed);

  virtual
  ~PowerLawFlowRule() {}
};


/**
 *  Thermally-activated flow rule.
 *
 *  \tparam ArgT The scalar type for computing the slip rate.
 */
template<typename ArgT>
struct ThermalActivationFlowRule final : public FlowRuleBase<ArgT>
{
  virtual
  ArgT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ArgT const & shear,
    ArgT const & slip_resistance,
    bool & failed);

  virtual
  ~ThermalActivationFlowRule() {}
};


/**
 *  Power Law with Viscous Drag flow rule.
 *
 *  \tparam ArgT The scalar type for computing the slip rate.
 */
template<typename ArgT>
struct PowerLawDragFlowRule final : public FlowRuleBase<ArgT>
{
  virtual
  ArgT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ArgT const & shear,
    ArgT const & slip_resistance,
    bool & failed);

  virtual
  ~PowerLawDragFlowRule() {}
};


/**
 *  Flow rule for no flow (elasticity)
 *
 *  \tparam ArgT The scalar type for computing the slip rate.
 */
template<typename ArgT>
struct NoFlowRule final : public FlowRuleBase<ArgT>
{
  virtual
  ArgT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ArgT const & shear,
    ArgT const & slip_resistance,
    bool & failed);

  virtual
  ~NoFlowRule() {}
};



}

#include "FlowRule_Def.hpp"

#endif
