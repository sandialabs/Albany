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

namespace LCM
{

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
  ~CrystalPlasticityModel()
  {
  }
  ;

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

  ///
  /// explicit update of the slip
  ///
  template<typename ArgT>
  void
  updateSlipViaExplicitIntegration(ScalarT dt,
      std::vector<ScalarT> const & slip_n,
      std::vector<ScalarT> const & hardness,
      Intrepid::Tensor<ArgT> const & S,
      std::vector<ArgT> const & shear,
      std::vector<ArgT> & slip_np1) const;

  ///
  /// Compute Lp_np1 and Fp_np1 based on computed slip increment
  ///
  template<typename ArgT>
  void
  applySlipIncrement(std::vector<ScalarT> const & slip_n,
      std::vector<ArgT> const & slip_np1,
      Intrepid::Tensor<ScalarT> const & Fp_n,
      Intrepid::Tensor<ArgT> & Lp_np1,
      Intrepid::Tensor<ArgT> & Fp_np1) const;

  ///
  /// update the hardness
  ///
  template<typename ArgT>
  void
  updateHardness(std::vector<ArgT> const & slip_np1,
      std::vector<ScalarT> const & hardness_n,
      std::vector<ArgT> & hardness_np1) const;

  ///
  /// residual
  ///
  template<typename ArgT>
  void
  computeResidual(ScalarT dt,
      std::vector<ScalarT> const & slip_n,
      std::vector<ArgT> const & slip_np1,
      std::vector<ArgT> const & hardness_np1,
      std::vector<ArgT> const & shear_np1,
      std::vector<ArgT> & slip_residual,
      ArgT & norm_slip_residual) const;

  ///
  /// compute stresses
  ///
  template<typename ArgT>
  void
  computeStress(Intrepid::Tensor<ScalarT> const & F,
      Intrepid::Tensor<ArgT> const & Fp,
      Intrepid::Tensor<ArgT> & T,
      Intrepid::Tensor<ArgT> & S,
      std::vector<ArgT> & shear) const;

  template<typename ArgT>
  void
  constructMatrixFiniteDifference(ScalarT dt,
      Intrepid::Tensor<ScalarT> const & Fp_n,
      Intrepid::Tensor<ScalarT> const & F_np1,
      std::vector<ScalarT> const & slip_n,
      std::vector<ArgT> const & slip_np1,
      std::vector<ScalarT> const & hardness_n,
      std::vector<ArgT> & matrix) const;

  template<typename ArgT>
  void
  lineSearch(ScalarT dt,
      Intrepid::Tensor<ScalarT> const & Fp_n,
      Intrepid::Tensor<ScalarT> const & F_np1,
      std::vector<ScalarT> const & slip_n,
      std::vector<ArgT> const & slip_np1_km1,
      std::vector<ArgT> const & delta_delta_slip,
      std::vector<ScalarT> const & hardness_n,
      ScalarT const & norm_slip_residual,
      RealType & alpha) const;

  ///
  /// Check tensor for nans and infs.
  ///
  template<typename ArgT>
  void
  confirmTensorSanity(Intrepid::Tensor<ArgT> const & input,
      std::string const & message) const;

  ///
  /// Crystal elasticity parameters
  ///
  RealType c11_, c12_, c44_;
  Intrepid::Tensor4<RealType> C_;
  Intrepid::Tensor<RealType> orientation_;

  ///
  /// Number of slip systems
  ///
  int num_slip_;

  //! \brief Struct to slip system information
  struct SlipSystemStruct {

    SlipSystemStruct()
    {
    }

    // slip system vectors
    Intrepid::Vector<RealType> s_, n_;

    // Schmid Tensor
    Intrepid::Tensor<RealType> projector_;

    // flow rule parameters
    RealType tau_critical_, gamma_dot_0_, gamma_exp_, H_, Rd_;
  };

  ///
  /// Crystal Plasticity parameters
  ///
  std::vector<SlipSystemStruct> slip_systems_;

  IntegrationScheme integration_scheme_;
  RealType implicit_nonlinear_solver_relative_tolerance_;
  RealType implicit_nonlinear_solver_absolute_tolerance_;
  int implicit_nonlinear_solver_max_iterations_;
};
}

#endif
