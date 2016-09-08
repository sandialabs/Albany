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
};

//
// Slip system family - collection of slip systems grouped by flow and
// hardening characteristics
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class SlipFamily
{
public:
  SlipFamily();

  ~SlipFamily() {}
  
  void setHardeningLawType(HardeningLawType law);
  HardeningLawType getHardeningLawType() const { return type_hardening_law_; }
  void setFlowRuleType(FlowRuleType rule);
  FlowRuleType getFlowRuleType() const { return type_flow_rule_; }

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

private:

  HardeningLawType
  type_hardening_law_{HardeningLawType::UNDEFINED};

  FlowRuleType
  type_flow_rule_{FlowRuleType::UNDEFINED};
};


template<typename ScalarT, Intrepid2::Index NumDimT>
struct StateMechanical
{
  using TensorType = Intrepid2::Tensor<ScalarT, NumDimT>;
  using InputTensorType = Intrepid2::Tensor<RealType, NumDimT>;

  StateMechanical(int num_dim, InputTensorType const & Fp_n)
    : num_dim_(num_dim),
      Fp_n_(Fp_n),
      Fp_np1_(num_dim),
      Lp_np1_(num_dim),
      sigma_np1_(num_dim),
      S_np1_(num_dim)
  {}

  int
  num_dim_;

  InputTensorType const
  Fp_n_;

  TensorType
  Fp_np1_;

  TensorType
  Lp_np1_;

  TensorType 
  sigma_np1_;

  TensorType
  S_np1_;
};


template<typename ScalarT, Intrepid2::Index NumSlipT>
struct StateInternal
{
  using VectorType = Intrepid2::Vector<ScalarT, NumSlipT>;
  using InputVectorType = Intrepid2::Vector<RealType, NumSlipT>;

  StateInternal(int num_slip, InputVectorType const & hardening_n,
      InputVectorType const & slip_n)
    : num_slip_(num_slip),
      hardening_n_(hardening_n),
      slip_n_(slip_n),
      rate_slip_(num_slip),
      hardening_np1_(num_slip),
      slip_np1_(num_slip),
      shear_np1_(num_slip),
      resistance_(num_slip)
  {}

  int
  num_slip_;

  InputVectorType const
  hardening_n_;

  InputVectorType const
  slip_n_;

  VectorType
  rate_slip_;

  VectorType
  hardening_np1_;

  VectorType
  slip_np1_;

  VectorType
  shear_np1_;
  
  VectorType
  resistance_;
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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
computeStress(
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    Intrepid2::Tensor4<ArgT, NumDimT> const & C,
    Intrepid2::Tensor<ArgT, NumDimT> const & F,
    Intrepid2::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear);



//
//! Construct elasticity tensor
//
template<Intrepid2::Index NumDimT, typename DataT, typename ArgT>
void
computeElasticityTensor(
    DataT c11, 
    DataT c12,
    DataT c13,
    DataT c33,
    DataT c44,
    DataT c66,
    Intrepid2::Tensor4<ArgT, NumDimT> & C);

} // namespace CP

#include "CrystalPlasticityCore_Def.hpp"

#endif
