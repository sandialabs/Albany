//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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

  std::map<std::string, ParamIndex>
  param_map_;

  // Flow parameters
  Intrepid2::Vector<RealType>
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
    flow_params_.fill(Intrepid2::ZEROS);
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


/**
 *	Parameters for the Power Law with Viscous Drag flow rule.
 */
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
    param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    param_map_["Rate Exponent"] = EXPONENT_RATE;
    param_map_["Drag Coefficient"] = DRAG_COEFF;
    flow_params_.set_dimension(NUM_PARAMS);
    flow_params_.fill(Intrepid2::ZEROS);
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
};


/**
 *  Base class for flow rules.
 *
 *  \tparam ScalarT The scalar type for computing the slip rate.
 */
template<typename ScalarT>
struct FlowRuleBase
{
  FlowRuleBase() {}

  virtual
  ScalarT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance) = 0;

  virtual
  ~FlowRuleBase() {}
};


/**
 *  Power Law flow rule.
 *
 *  \tparam ScalarT The scalar type for computing the slip rate.
 */
template<typename ScalarT>
struct PowerLawFlowRule final : public FlowRuleBase<ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~PowerLawFlowRule() {}
};


/**
 *  Thermally-activated flow rule.
 *
 *  \tparam ScalarT The scalar type for computing the slip rate.
 */
template<typename ScalarT>
struct ThermalActivationFlowRule final : public FlowRuleBase<ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~ThermalActivationFlowRule() {}
};


/**
 *  Power Law with Viscous Drag flow rule.
 *
 *  \tparam ScalarT The scalar type for computing the slip rate.
 */
template<typename ScalarT>
struct PowerLawDragFlowRule final : public FlowRuleBase<ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~PowerLawDragFlowRule() {}
};


/**
 *  Flow rule for no flow (elasticity)
 *
 *  \tparam ScalarT The scalar type for computing the slip rate.
 */
template<typename ScalarT>
struct NoFlowRule final : public FlowRuleBase<ScalarT>
{
  virtual
  ScalarT
  computeRateSlip(
    std::shared_ptr<FlowParameterBase> const & pflow_parameters,
    ScalarT const & shear,
    ScalarT const & slip_resistance);

  virtual
  ~NoFlowRule() {}
};



}

#include "FlowRule_Def.hpp"

#endif
