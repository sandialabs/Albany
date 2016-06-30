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

enum class FlowRule
{
  UNDEFINED = 0,
  POWER_LAW = 1,
  THERMAL_ACTIVATION = 2
};

enum class HardeningLaw
{
  UNDEFINED = 0, 
  EXPONENTIAL = 1, 
  SATURATION = 2, 
  DISLOCATION_DENSITY = 3
};



//
//! Struct containing slip system information.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct SlipSystemStruct
{

  SlipSystemStruct() {}

  //! Slip system vectors.
  Intrepid2::Vector<RealType, NumDimT>
  s_;

  Intrepid2::Vector<RealType, NumDimT>
  n_;

  //! Schmid Tensor.
  Intrepid2::Tensor<RealType, NumDimT> 
  projector_;

  //
  // Flow rule parameters
  //
  FlowRule 
  flow_rule;

  RealType 
  rate_slip_reference_;

  RealType
  exponent_rate_;

  RealType
  energy_activation_;

  //
  // Hardening law parameters
  //
  HardeningLaw
  hardening_law;

  ///
  /// Hardening parameters: linear hardening
  ///
  RealType
  tau_critical_;

  RealType
  H_;

  RealType
  Rd_;

  ///
  /// Hardening parameters: saturation hardening
  ///    
  RealType
  resistance_slip_initial_;

  RealType
  rate_hardening_;

  RealType
  stress_saturation_initial_;

  RealType
  exponent_saturation_;

  ///
  /// Hardening parameters: dislocation density hardening
  /// 
  RealType
  factor_geometry_dislocation_;

  RealType
  density_dislocation_;

  RealType
  c_generation_;

  RealType
  c_annihilation_;

  RealType
  modulus_shear_;

  RealType
  magnitude_burgers_;

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
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
applySlipIncrement(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp_n,
    Intrepid2::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid2::Tensor<ArgT, NumDimT> & Fp_np1);



//
//! Update the hardness.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);



///
/// Update the plastic slips
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
updateSlip(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_resistance,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1);



//
//! Compute stress.
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT, typename DataS>
void
computeStress(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid2::Tensor4<DataS, NumDimT> const & C,
    Intrepid2::Tensor<DataS, NumDimT> const & F,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp,
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
class CrystalPlasticityNLS:
    public Intrepid2::Function_Base<
    CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  CrystalPlasticityNLS(
      Intrepid2::Tensor4<ArgT, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
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

  Intrepid2::Tensor4<ArgT, NumDimT> const &
  C_;

  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const &
  slip_systems_;

  Intrepid2::Tensor<RealType, NumDimT> const &
  Fp_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  state_hardening_n_;

  Intrepid2::Vector<RealType, NumSlipT> const &
  slip_n_;

  Intrepid2::Tensor<ArgT, NumDimT> const &
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
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  ResidualSlipHardnessNLS(
      Intrepid2::Tensor4<ArgT, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
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

  RealType num_dim_;
  RealType num_slip_;
  Intrepid2::Tensor4<ArgT, NumDimT> const & C_;
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems_;
  Intrepid2::Tensor<RealType, NumDimT> const & Fp_n_;
  Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n_;
  Intrepid2::Vector<RealType, NumSlipT> const & slip_n_;
  Intrepid2::Tensor<ArgT, NumDimT> const & F_np1_;
  RealType dt_;
};

///
/// Hardening Base
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
struct HardeningBase
{
  HardeningBase() {}

  virtual
  char const * const
  name() = 0;

  virtual
  void
  createLatentMatrix(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems) = 0;

  virtual
  void
  harden(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance) = 0;

  virtual
  ~HardeningBase() {}
};

///
///
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
std::unique_ptr<HardeningBase<NumDimT, NumSlipT, DataT, ArgT>>
HardeningFactory(HardeningLaw hardening_law);

///
/// Linear hardening with recovery
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
struct LinearMinusRecoveryHardening final : public HardeningBase<NumDimT, NumSlipT, DataT, ArgT>
{
  static constexpr
  char const * const
  NAME{"Exponential"};

  virtual
  char const * const
  name()
  {
    return NAME;
  }

  virtual
  void
  createLatentMatrix(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems);

  virtual
  void
  harden(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~LinearMinusRecoveryHardening() {}

private:
  Intrepid2::Tensor<DataT, NumSlipT>
  latent_matrix;//{Intrepid2::identity<DataT, NumSlipT>(NumSlipT)};


};

///
/// Saturation hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
struct SaturationHardening final : public HardeningBase<NumDimT, NumSlipT, DataT, ArgT>
{
  static constexpr
  char const * const
  NAME{"Saturation"};

  virtual
  char const * const
  name()
  {
    return NAME;
  }

  virtual
  void
  createLatentMatrix(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems);

  virtual
  void
  harden(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~SaturationHardening() {}

private:
  Intrepid2::Tensor<DataT, NumSlipT>
  latent_matrix;//{Intrepid2::identity<DataT, NumSlipT>(NumSlipT)};

};

///
/// Dislocation-density based hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
struct DislocationDensityHardening final : public HardeningBase<NumDimT, NumSlipT, DataT, ArgT>
{
  static constexpr
  char const * const
  NAME{"Dislocation-Density Based"};

  virtual
  char const * const
  name()
  {
    return NAME;
  }

  virtual
  void
  createLatentMatrix(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems);

  virtual
  void
  harden(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~DislocationDensityHardening() {}

private:
  Intrepid2::Tensor<DataT, NumSlipT>
  latent_matrix;//{Intrepid2::identity<DataT, NumSlipT>(NumSlipT)};

};

///
/// No hardening
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
struct NoHardening final : public HardeningBase<NumDimT, NumSlipT, DataT, ArgT>
{
  static constexpr
  char const * const
  NAME{"No hardening"};

  virtual
  char const * const
  name()
  {
    return NAME;
  }

  virtual
  void
  createLatentMatrix(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems);

  virtual
  void
  harden(
    std::vector<SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~NoHardening() {}

private:
  Intrepid2::Tensor<DataT, NumSlipT>
  latent_matrix;//{Intrepid2::identity<DataT, NumSlipT>(NumSlipT)};

};

//
//
//
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, 
  typename DataT, typename ArgT>
std::unique_ptr<HardeningBase<NumDimT, NumSlipT, DataT, ArgT>>
hardeningFactory(HardeningLaw hardening_law)
{
  using HTUP = std::unique_ptr<HardeningBase<NumDimT, NumSlipT, DataT, ArgT>>;

  switch (hardening_law) {

  default:
    std::cerr << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "ERROR: Unknown hardening law\n";
    exit(1);
    break;

  case HardeningLaw::EXPONENTIAL:
    return HTUP(new LinearMinusRecoveryHardening<NumDimT, NumSlipT, DataT, ArgT>());
    break;

  case HardeningLaw::SATURATION:
    return HTUP(new SaturationHardening<NumDimT, NumSlipT, DataT, ArgT>());
    break;

  case HardeningLaw::DISLOCATION_DENSITY:
    return HTUP(new DislocationDensityHardening<NumDimT, NumSlipT, DataT, ArgT>());
    break;

  case HardeningLaw::UNDEFINED:
    return HTUP(new NoHardening<NumDimT, NumSlipT, DataT, ArgT>());
    break;

  }

  return HTUP(nullptr);
}

} // namespace CP

#include "CrystalPlasticityCore_Def.hpp"

#endif
