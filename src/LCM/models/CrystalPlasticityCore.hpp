//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(CrystalPlasticityCore_hpp)
#define CrystalPlasticityCore_hpp

#include <Intrepid_MiniTensor.h>
#include <MiniNonlinearSolver.h>

namespace CP
{

static constexpr Intrepid::Index MAX_NUM_DIM = 3;
static constexpr Intrepid::Index MAX_NUM_SLIP = 12;

//! Struct containing slip system information.
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT>
struct SlipSystemStruct
{

  SlipSystemStruct() {}

  //! Slip system vectors.
  Intrepid::Vector<RealType, NumDimT> s_, n_;

  //! Schmid Tensor.
  Intrepid::Tensor<RealType, NumDimT> projector_;

  //! Flow rule parameters.
  RealType tau_critical_, gamma_dot_0_, gamma_exp_, H_, Rd_;
};

//! Check tensor for NaN and inf values.
template<Intrepid::Index NumDimT, typename ArgT>
void
confirmTensorSanity(
    Intrepid::Tensor<ArgT, NumDimT> const & input,
    std::string const & message);

//! Compute Lp_np1 and Fp_np1 based on computed slip increment.
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename DataT,
    typename ArgT>
void
applySlipIncrement(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid::Tensor<DataT, NumDimT> const & Fp_n,
    Intrepid::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid::Tensor<ArgT, NumDimT> & Fp_np1);

//! Update the hardness.
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename DataT,
    typename ArgT>
void
updateHardness(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid::Vector<DataT, NumSlipT> const & hardness_n,
    Intrepid::Vector<ArgT, NumSlipT> & hardness_np1);

//! Evaluate the slip residual.
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename DataT,
    typename ArgT>
void
computeResidual(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid::Vector<ArgT, NumSlipT> const & hardness_np1,
    Intrepid::Vector<ArgT, NumSlipT> const & shear_np1,
    Intrepid::Vector<ArgT, NumSlipT> & slip_residual,
    ArgT & norm_slip_residual);

//! Compute stress.
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename DataT,
    typename ArgT>
void
computeStress(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    Intrepid::Tensor4<RealType, NumDimT> const & C,
    Intrepid::Tensor<DataT, NumDimT> const & F,
    Intrepid::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid::Tensor<ArgT, NumDimT> & sigma,
    Intrepid::Tensor<ArgT, NumDimT> & S,
    Intrepid::Vector<ArgT, NumSlipT> & shear);

//! Update the slip via explicit integration (explicit state update).
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename DataT,
    typename ArgT>
void
updateSlipViaExplicitIntegration(
    std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
    DataT dt,
    Intrepid::Vector<DataT, NumSlipT> const & slip_n,
    Intrepid::Vector<DataT, NumSlipT> const & hardness,
    Intrepid::Tensor<ArgT, NumDimT> const & S,
    Intrepid::Vector<ArgT, NumSlipT> const & shear,
    Intrepid::Vector<ArgT, NumSlipT> & slip_np1);

//! Base class for recording the dimension of a Nonlinear Solver (NLS) class, required because templates..
class NLSDimension
{
public:
  static Intrepid::Index DIMENSION;
};

//! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip increments as unknowns.
template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename EvalT>
class CrystalPlasticityNLS:
    public NLSDimension,
    public Intrepid::Function_Base<
    CrystalPlasticityNLS<NumDimT, NumSlipT, EvalT>, typename EvalT::ScalarT>
{
  using DataT = typename EvalT::ScalarT;

public:

  //! Constructor.
  CrystalPlasticityNLS(
      Intrepid::Tensor4<RealType, NumDimT> const & C,
      std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid::Tensor<RealType, NumDimT> const & Fp_n,
      Intrepid::Vector<RealType, NumSlipT> const & hardness_n,
      Intrepid::Vector<RealType, NumSlipT> const & slip_n,
      Intrepid::Tensor<DataT, NumDimT> const & F_np1,
      RealType dt);

  static constexpr char const * const NAME =
      "Crystal Plasticity Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  T
  value(Intrepid::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the slip at step N+1.
  template<typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  gradient(Intrepid::Vector<T, N> const & slip_np1) const;

  //! Default implementation of hessian.
  template<typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Tensor<T, N>
  hessian(Intrepid::Vector<T, N> const & x);

private:

  RealType num_dim_;
  RealType num_slip_;
  Intrepid::Tensor4<RealType, NumDimT> const & C_;
  std::vector<CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems_;
  Intrepid::Tensor<RealType, NumDimT> const & Fp_n_;
  Intrepid::Vector<RealType, NumSlipT> const & hardness_n_;
  Intrepid::Vector<RealType, NumSlipT> const & slip_n_;
  Intrepid::Tensor<DataT, NumDimT> const & F_np1_;
  RealType dt_;
};

} // namespace CP

#include "CrystalPlasticityCore_Def.hpp"

#endif
