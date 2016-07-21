//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(CrystalPlasticityCore_hpp)
#define CrystalPlasticityCore_hpp

#include <MiniNonlinearSolver.h>
#include "CrystalPlasticityFwd.hpp"
#include "FlowRule.hpp"
#include "HardeningLaw.hpp"

namespace CP
{

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

  std::shared_ptr<HardeningParameterBase<NumDimT, NumSlipT>>
  phardening_parameters_{nullptr};

  std::shared_ptr<FlowParameterBase>
  pflow_parameters_{nullptr};

  Intrepid2::Tensor<RealType, NumSlipT>
  latent_matrix_;

  HardeningLawType
  type_hardening_law_{HardeningLawType::UNDEFINED};

  FlowRuleType
  type_flow_rule_{FlowRuleType::UNDEFINED};
};



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
// Factory returning a pointer to a flow rule object
//
template<typename ScalarT>
class flowRuleFactory
{

public:

  using FlowRuleBaseType = FlowRuleBase<ScalarT>;

  FlowRuleBaseType *
  createFlowRule(FlowRuleType type_flow_rule) {  

    switch (type_flow_rule) {

    default:
      std::cerr << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "ERROR: Unknown flow rule\n";
      exit(1);
      break;

    case FlowRuleType::POWER_LAW:
      return new(flow_buffer_) PowerLawFlowRule<ScalarT>();
      break;

    case FlowRuleType::POWER_LAW_DRAG:
      return new(flow_buffer_) PowerLawDragFlowRule<ScalarT>();
      break;

    case FlowRuleType::THERMAL_ACTIVATION:
      return new(flow_buffer_) ThermalActivationFlowRule<ScalarT>();
      break;

    case FlowRuleType::UNDEFINED:
      return new(flow_buffer_) NoFlowRule<ScalarT>();
      break;
    }

    return nullptr;
  }

private:

  unsigned char
  flow_buffer_[sizeof(FlowRuleBaseType)];
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
