//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(CrystalPlasticityCore_hpp)
#define CrystalPlasticityCore_hpp

#include <MiniNonlinearSolver.h>

namespace CP
{

static constexpr Intrepid2::Index MAX_DIM = 3;
static constexpr Intrepid2::Index MAX_SLIP = 12;

//! Struct containing slip system information.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct SlipSystemStruct
{

  SlipSystemStruct() {}

  //! Slip system vectors.
  Intrepid2::Vector<RealType, NumDimT> s_, n_;

  //! Schmid Tensor.
  Intrepid2::Tensor<RealType, NumDimT> projector_;

  //! Flow rule parameters
  int flow_rule;
  RealType rate_slip_reference_, exponent_rate_, energy_activation_;

  // hardening law parameters
  int hardening_law;
  RealType tau_critical_, H_, Rd_, resistance_slip_initial_,
    rate_hardening_, stress_saturation_initial_, exponent_saturation_;

};

//! Check tensor for NaN and inf values.
template<Intrepid2::Index NumDimT, typename ArgT>
void
confirmTensorSanity(
    Intrepid2::Tensor<ArgT, NumDimT> const & input,
    std::string const & message);

//! Compute Lp_np1 and Fp_np1 based on computed slip increment.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
applySlipIncrement(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp_n,
    Intrepid2::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid2::Tensor<ArgT, NumDimT> & Fp_np1);

//! Update the hardness.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<DataT, NumSlipT> const & hardness_n,
    Intrepid2::Vector<ArgT, NumSlipT> & hardness_np1);

//! Evaluate the slip residual.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
computeResidual(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Vector<ArgT, NumSlipT> const & hardness_np1,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_residual,
    ArgT & norm_slip_residual);

//! Compute stress.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
computeStress(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid2::Tensor4<RealType, NumDimT> const & C,
    Intrepid2::Tensor<ArgT, NumDimT> const & F,
    Intrepid2::Tensor<DataT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear);

//! Update the slip via explicit integration (explicit state update).
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename DataT,
    typename ArgT>
void
updateSlipViaExplicitIntegration(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid2::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & hardness,
    Intrepid2::Tensor<ArgT, NumDimT> const & S,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1);


//! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip 
//  increments as unknowns.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
class CrystalPlasticityNLS:
    public Intrepid2::Function_Base<
    CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  CrystalPlasticityNLS(
      Intrepid2::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
      RealType dt);

  static constexpr char const * const NAME =
      "Crystal Plasticity Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the slip at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const;


  //! Default implementation of hessian.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

private:

  RealType num_dim_;
  RealType num_slip_;
  Intrepid2::Tensor4<RealType, NumDimT> const & C_;
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems_;
  Intrepid2::Tensor<RealType, NumDimT> const & Fp_n_;
  Intrepid2::Vector<RealType, NumSlipT> const & hardness_n_;
  Intrepid2::Vector<RealType, NumSlipT> const & slip_n_;
  Intrepid2::Tensor<ArgT, NumDimT> const & F_np1_;
  RealType dt_;
};

//! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip 
//  increments and hardnesses as unknowns.
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
class ResidualSlipHardnessNLS:
    public Intrepid2::Function_Base<
    ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  ResidualSlipHardnessNLS(
      Intrepid2::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid2::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid2::Tensor<ArgT, NumDimT> const & F_np1,
      RealType dt);

  static constexpr char const * const NAME =
      "Slip and Hardness Residual Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the slip at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const;


  //! Default implementation of hessian.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

private:

  RealType num_dim_;
  RealType num_slip_;
  Intrepid2::Tensor4<RealType, NumDimT> const & C_;
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems_;
  Intrepid2::Tensor<RealType, NumDimT> const & Fp_n_;
  Intrepid2::Vector<RealType, NumSlipT> const & hardness_n_;
  Intrepid2::Vector<RealType, NumSlipT> const & slip_n_;
  Intrepid2::Tensor<ArgT, NumDimT> const & F_np1_;
  RealType dt_;
};

} // namespace CP

#include "CrystalPlasticityCore_Def.hpp"

#endif
