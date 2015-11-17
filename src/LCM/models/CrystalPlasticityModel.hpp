//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_hpp)
#define LCM_CrystalPlasticityModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include <Intrepid_MiniTensor.h>
#include "Intrepid_MiniTensor_Solvers.h"

namespace LCM
{

namespace CP
{

  static constexpr Intrepid::Index MAX_NUM_DIM = 3;
  static constexpr Intrepid::Index MAX_NUM_SLIP = 12;

  //! \brief Struct to slip system information
  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT>
  struct SlipSystemStruct {

    SlipSystemStruct() {}

    // slip system vectors
    Intrepid::Vector<RealType, NumDimT> s_, n_;

    // Schmid Tensor
    Intrepid::Tensor<RealType, NumDimT> projector_;

    // flow rule parameters
    RealType tau_critical_, gamma_dot_0_, gamma_exp_, H_, Rd_;
  };

  // Check tensor for NaN and inf values.
  template<Intrepid::Index NumDimT, typename ArgT>
  void
  confirmTensorSanity(
      Intrepid::Tensor<ArgT, NumDimT> const & input,
      std::string const & message);

  // Compute Lp_np1 and Fp_np1 based on computed slip increment
  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT, typename ArgT>
  void
  applySlipIncrement(
      std::vector< CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
      Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
      Intrepid::Tensor<ScalarT, NumDimT> const & Fp_n,
      Intrepid::Tensor<ArgT, NumDimT> & Lp_np1,
      Intrepid::Tensor<ArgT, NumDimT> & Fp_np1);

  // Update the hardness
  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT, typename ArgT>
  void
  updateHardness(
      std::vector< CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
      Intrepid::Vector<ScalarT, NumSlipT> const & hardness_n,
      Intrepid::Vector<ArgT, NumSlipT> & hardness_np1);

  /// Evaluate the slip residual
  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT, typename ArgT>
  void
  computeResidual(
      std::vector< CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      ScalarT dt,
      Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
      Intrepid::Vector<ArgT, NumSlipT> const & slip_np1,
      Intrepid::Vector<ArgT, NumSlipT> const & hardness_np1,
      Intrepid::Vector<ArgT, NumSlipT> const & shear_np1,
      Intrepid::Vector<ArgT, NumSlipT> & slip_residual,
      ArgT & norm_slip_residual);

  /// Compute stress
  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT, typename ArgT>
  void
  computeStress(
      std::vector< CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
      Intrepid::Tensor4<RealType, NumDimT> const & C,
      Intrepid::Tensor<ScalarT, NumDimT> const & F,
      Intrepid::Tensor<ArgT, NumDimT> const & Fp,
      Intrepid::Tensor<ArgT, NumDimT> & T,
      Intrepid::Tensor<ArgT, NumDimT> & S,
      Intrepid::Vector<ArgT, NumSlipT> & shear);

}

template <Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ScalarT>
class CrystalPlasticityNLS : public Intrepid::Function_Base<CrystalPlasticityNLS<NumDimT, NumSlipT, ScalarT>, ScalarT>
{
public:

  CrystalPlasticityNLS(Intrepid::Tensor4<RealType, NumDimT> const & C,
		       std::vector< CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems,
		       Intrepid::Tensor<RealType, NumDimT> const & Fp_n,
		       Intrepid::Vector<RealType, NumSlipT> const & hardness_n,
		       Intrepid::Vector<RealType, NumSlipT> const & slip_n,
		       Intrepid::Tensor<ScalarT, NumDimT> const & F_np1,
		       RealType dt)
    : C_(C), slip_systems_(slip_systems), Fp_n_(Fp_n), hardness_n_(hardness_n),
      slip_n_(slip_n), F_np1_(F_np1), dt_(dt)
  {
    num_dim_ = Fp_n_.get_dimension();
    num_slip_ = hardness_n_.get_dimension();
  }

  static constexpr Intrepid::Index DIMENSION = CP::MAX_NUM_SLIP;
  static constexpr char const * const NAME = "Crystal Plasticity Nonlinear System";

  // Default value.
  template<typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  T
  value(Intrepid::Vector<T, N> const & x)
  {
    return Intrepid::Function_Base<CrystalPlasticityNLS<NumDimT, NumSlipT, ScalarT>, ScalarT>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  gradient(Intrepid::Vector<T, N> const & slip_np1) const
  {
    // DJL todo: Experiment with how/where these are allocated.
    Intrepid::Tensor<T, NumDimT> Fp_np1;
    Intrepid::Tensor<T, NumDimT> Lp_np1;
    Intrepid::Vector<T, N> hardness_np1;
    Intrepid::Tensor<T, NumDimT> sigma_np1;
    Intrepid::Tensor<T, NumDimT> S_np1;
    Intrepid::Vector<T, N> shear_np1;
    Intrepid::Vector<T, N> slip_residual;
    T norm_slip_residual_;

    Fp_np1.set_dimension(num_dim_);
    Lp_np1.set_dimension(num_dim_);
    hardness_np1.set_dimension(num_slip_);
    sigma_np1.set_dimension(num_slip_);
    S_np1.set_dimension(num_dim_);
    shear_np1.set_dimension(num_slip_);
    slip_residual.set_dimension(num_slip_);

    // DJL todo:
    //T const F_np1_hope = convert<ScalarT, T>(F_np1__);

    // Compute Lp_np1, and Fp_np1
    CP::applySlipIncrement<NumDimT, NumSlipT>(slip_systems_, slip_n_, slip_np1, Fp_n_, Lp_np1, Fp_np1);

    // Compute hardness_np1
    CP::updateHardness<NumDimT, NumSlipT>(slip_systems_, slip_np1, hardness_n_, hardness_np1);

    // Compute sigma_np1, S_np1, and shear_np1
    CP::computeStress<NumDimT, NumSlipT>(slip_systems_, C_, F_np1_, Fp_np1, sigma_np1, S_np1, shear_np1);

    // Compute slip_residual and norm_slip_residual
    CP::computeResidual<NumDimT, NumSlipT>(slip_systems_, dt_, slip_n_, slip_np1, hardness_np1, shear_np1, slip_residual, norm_slip_residual_);

    return slip_residual;
  }

  // Default AD hessian.
  template<typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Tensor<T, N>
  hessian(Intrepid::Vector<T, N> const & x)
  {
    return Intrepid::Function_Base<CrystalPlasticityNLS<NumDimT, NumSlipT, ScalarT>, ScalarT>::hessian(*this, x);
  }

private:

  RealType num_dim_;
  RealType num_slip_;
  Intrepid::Tensor4<RealType, NumDimT> const & C_;
  std::vector< CP::SlipSystemStruct<NumDimT, NumSlipT> > const & slip_systems_;
  Intrepid::Tensor<RealType, NumDimT> const & Fp_n_;
  Intrepid::Vector<RealType, NumSlipT> const & hardness_n_;
  Intrepid::Vector<RealType, NumSlipT> const & slip_n_;
  Intrepid::Tensor<ScalarT, NumDimT> const & F_np1_;
  RealType dt_;
};

//! \brief CrystalPlasticity Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class CrystalPlasticityModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  enum IntegrationScheme {
    EXPLICIT = 0, IMPLICIT = 1
  };

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // typedef for automatic differentiation type used in internal Newton loop
  // options are:  DFad (dynamically sized), SFad (static size), SLFad (bounded)
//   typedef typename Sacado::Fad::DFad<ScalarT> Fad;
  typedef typename Sacado::Fad::SLFad<ScalarT, 12> Fad;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  ///
  /// Constructor
  ///
  CrystalPlasticityModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~CrystalPlasticityModel() {}

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

private:

  ///
  /// Private to prohibit copying
  ///
  CrystalPlasticityModel(const CrystalPlasticityModel&);

  ///
  /// Private to prohibit copying
  ///
  CrystalPlasticityModel& operator=(const CrystalPlasticityModel&);

  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ArgT>
  void
  lineSearch(
      ScalarT dt,
      Intrepid::Tensor<ScalarT, NumDimT> const & Fp_n,
      Intrepid::Tensor<ScalarT, NumDimT> const & F_np1,
      Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
      Intrepid::Vector<ArgT, NumSlipT> const & slip_np1_km1,
      Intrepid::Vector<ArgT, NumSlipT> const & delta_delta_slip,
      Intrepid::Vector<ScalarT, NumSlipT> const & hardness_n,
      ScalarT const & norm_slip_residual,
      RealType & alpha) const;

  ///
  /// explicit update of the slip
  ///
  template<Intrepid::Index NumDimT, Intrepid::Index NumSlipT, typename ArgT>
  void
  updateSlipViaExplicitIntegration(
      ScalarT dt,
      Intrepid::Vector<ScalarT, NumSlipT> const & slip_n,
      Intrepid::Vector<ScalarT, NumSlipT> const & hardness,
      Intrepid::Tensor<ArgT, NumDimT> const & S,
      Intrepid::Vector<ArgT, NumSlipT> const & shear,
      Intrepid::Vector<ArgT, NumSlipT> & slip_np1) const;

  ///
  /// Crystal elasticity parameters
  ///
  RealType c11_, c12_, c44_;
  Intrepid::Tensor4<RealType, CP::MAX_NUM_DIM> C_;
  Intrepid::Tensor<RealType, CP::MAX_NUM_DIM> orientation_;

  ///
  /// Number of slip systems
  ///
  int num_slip_;

  ///
  /// Crystal Plasticity parameters
  ///
  std::vector< CP::SlipSystemStruct<CP::MAX_NUM_DIM,CP::MAX_NUM_SLIP> > slip_systems_;

  IntegrationScheme integration_scheme_;
  RealType implicit_nonlinear_solver_relative_tolerance_;
  RealType implicit_nonlinear_solver_absolute_tolerance_;
  int implicit_nonlinear_solver_max_iterations_;
};
}

#endif
