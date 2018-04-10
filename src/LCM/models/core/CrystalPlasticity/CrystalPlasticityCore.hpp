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
template<minitensor::Index NumDimT>
struct SlipSystem
{
  SlipSystem() {}

  minitensor::Index
  slip_family_index_;

  //! Slip system vectors.
  minitensor::Vector<RealType, MAX_DIM>
  s_;

  minitensor::Vector<RealType, MAX_DIM>
  n_;

  //! Schmid Tensor.
  minitensor::Tensor<RealType, MAX_DIM>
  projector_;

  //
  RealType
  state_hardening_initial_;
};

//
// Slip system family - collection of slip systems grouped by flow and
// hardening characteristics
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
class SlipFamily
{
public:
  SlipFamily();

  ~SlipFamily() {}

  void setHardeningLawType(HardeningLawType law);
  HardeningLawType getHardeningLawType() const { return type_hardening_law_; }
  void setFlowRuleType(FlowRuleType rule);
  FlowRuleType getFlowRuleType() const { return type_flow_rule_; }

  minitensor::Index
  num_slip_sys_{0};

  minitensor::Vector<minitensor::Index, NumSlipT>
  slip_system_indices_;

  std::shared_ptr<HardeningParameterBase<NumDimT, NumSlipT>>
  phardening_parameters_{nullptr};

  std::shared_ptr<FlowParameterBase>
  pflow_parameters_{nullptr};

  minitensor::Tensor<RealType, NumSlipT>
  latent_matrix_;

  minitensor::Tensor<RealType, NumSlipT>
  aux_matrix_;

private:

  HardeningLawType
  type_hardening_law_{HardeningLawType::UNDEFINED};

  FlowRuleType
  type_flow_rule_{FlowRuleType::UNDEFINED};
};


template<typename ScalarT, minitensor::Index NumDimT>
struct StateMechanical
{
  using TensorType = minitensor::Tensor<ScalarT, NumDimT>;
  using InputTensorType = minitensor::Tensor<RealType, NumDimT>;

  StateMechanical(
    int num_dim,
    InputTensorType const & F_n,
    InputTensorType const & Fp_n,
    TensorType const & F_np1)
    : num_dim_(num_dim),
      F_n_(F_n),
      Fp_n_(Fp_n),
      F_np1_(F_np1),
      Fp_np1_(num_dim),
      Lp_np1_(num_dim),
      sigma_np1_(num_dim),
      S_np1_(num_dim)
  {}

  int
  num_dim_;

  InputTensorType const
  F_n_;

  InputTensorType const
  Fp_n_;

  TensorType const
  F_np1_;

  TensorType
  Fp_np1_;

  TensorType
  Lp_np1_;

  TensorType
  sigma_np1_;

  TensorType
  S_np1_;
};

template<typename ScalarT, minitensor::Index NumSlipT>
struct StateInternal
{
  using VectorType = minitensor::Vector<ScalarT, NumSlipT>;
  using InputVectorType = minitensor::Vector<RealType, NumSlipT>;

StateInternal(int cell, int pt, int num_slip, InputVectorType const & hardening_n,
      InputVectorType const & slip_n)
    : cell_(cell),
      pt_(pt),
      num_slip_(num_slip),
      hardening_n_(hardening_n),
      slip_n_(slip_n),
      rates_slip_(num_slip),
      hardening_np1_(num_slip),
      slip_np1_(num_slip),
      shear_np1_(num_slip),
      resistance_(num_slip)
  {
    rates_slip_.fill(minitensor::Filler::ZEROS);
    hardening_np1_.fill(minitensor::Filler::ZEROS);
    slip_np1_.fill(minitensor::Filler::ZEROS);
    shear_np1_.fill(minitensor::Filler::ZEROS);
    resistance_.fill(minitensor::Filler::ZEROS);
  }

  int
  cell_;

  int
  pt_;

  int
  num_slip_;

  InputVectorType const
  hardening_n_;

  InputVectorType const
  slip_n_;

  VectorType
  rates_slip_;

  VectorType
  hardening_np1_;

  VectorType
  slip_np1_;

  VectorType
  shear_np1_;

  VectorType
  resistance_;
};

///
/// Verify that constitutive update has preserved finite values
///
template<typename T, minitensor::Index N>
void
expectFiniteTensor(
                   minitensor::Tensor<T, N> const & A,
                   std::string const & msg);

//
//! Compute Lp_np1 and Fp_np1 based on computed slip increment.
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
applySlipIncrement(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    minitensor::Vector<RealType, NumSlipT> const & slip_n,
    minitensor::Vector<ArgT, NumSlipT> const & slip_np1,
    minitensor::Tensor<RealType, NumDimT> const & Fp_n,
    minitensor::Tensor<ArgT, NumDimT> & Lp_np1,
    minitensor::Tensor<ArgT, NumDimT> & Fp_np1);



//
//! Update the hardness.
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
updateHardness(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    minitensor::Vector<ArgT, NumSlipT> const & rate_slip,
    minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
    minitensor::Vector<ArgT, NumSlipT> & state_hardening_np1,
    minitensor::Vector<ArgT, NumSlipT> & slip_resistance,
    bool & failed);



///
/// Update the plastic slips
///
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
updateSlip(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    minitensor::Vector<ArgT, NumSlipT> const & slip_resistance,
    minitensor::Vector<ArgT, NumSlipT> const & shear,
    minitensor::Vector<RealType, NumSlipT> const & slip_n,
    minitensor::Vector<ArgT, NumSlipT> & slip_np1,
    bool & failed);

//
//! Compute stress.
//
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
computeStress(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    minitensor::Tensor4<ArgT, NumDimT> const & C,
    minitensor::Tensor<ArgT, NumDimT> const & F,
    minitensor::Tensor<ArgT, NumDimT> const & Fp,
    minitensor::Tensor<ArgT, NumDimT> & sigma,
    minitensor::Tensor<ArgT, NumDimT> & S,
    minitensor::Vector<ArgT, NumSlipT> & shear,
    bool & failed);

//
//! Construct elasticity tensor
//
template<minitensor::Index NumDimT, typename DataT, typename ArgT>
void
computeElasticityTensor(
    DataT c11,
    DataT c12,
    DataT c13,
    DataT c33,
    DataT c44,
    DataT c66,
    minitensor::Tensor4<ArgT, NumDimT> & C);

} // namespace CP

#include "CrystalPlasticityCore_Def.hpp"

#endif
