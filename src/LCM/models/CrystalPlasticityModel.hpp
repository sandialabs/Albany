//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_hpp)
#define LCM_CrystalPlasticityModel_hpp

#include "core/CrystalPlasticity/CrystalPlasticityCore.hpp"
#include "core/CrystalPlasticity/NonlinearSolver.hpp"
#include "core/CrystalPlasticity/Integrator.hpp"
#include "ParallelConstitutiveModel.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"
#include "../../utility/StaticAllocator.hpp"

namespace LCM
{

//! \brief CrystalPlasticity Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class CrystalPlasticityKernel: public ParallelKernel<EvalT, Traits>
{
public:

  using ScalarT = typename EvalT::ScalarT;
  using ValueT = typename Sacado::ValueType<ScalarT>::type;

  using BaseKernel = ParallelKernel<EvalT, Traits>;
  using ScalarField = typename BaseKernel::ScalarField;
  using ConstScalarField = typename BaseKernel::ConstScalarField;
  using Workset = typename BaseKernel::Workset;

  // Dimension of problem, e.g., 2 -> 2D, 3 -> 3D
  using BaseKernel::num_dims_;

  using BaseKernel::num_pts_;
  using BaseKernel::field_name_map_;

  // optional temperature support
  using BaseKernel::have_temperature_;
  using BaseKernel::expansion_coeff_;
  using BaseKernel::ref_temperature_;
  using BaseKernel::heat_capacity_;
  using BaseKernel::density_;
  // using BaseKernel::temperature_;

  /// Pointer to NOX status test, allows the material model to force a global load step reduction
  using BaseKernel::nox_status_test_;

  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;
  using BaseKernel::addStateVariable;
  using BaseKernel::extractEvaluatedFieldArray;

  using SSV = Sacado::ScalarValue<ScalarT>;

  ///
  /// Constructor
  ///
  CrystalPlasticityKernel(
      ConstitutiveModel<EvalT, Traits> & model,
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  CrystalPlasticityKernel(CrystalPlasticityKernel const &) = delete;
  CrystalPlasticityKernel & operator=(CrystalPlasticityKernel const &) = delete;

  ///
  /// Virtual Deconstructor
  ///
  virtual
  ~CrystalPlasticityKernel()
  {
  }

  void
  init(Workset & workset,
       FieldMap<const ScalarT> & dep_fields,
       FieldMap<ScalarT> & eval_fields);

  ///
  /// Method to compute the state for a single cell and quadrature point
  //  (e.g. energy, stress, tangent)
  ///
  KOKKOS_INLINE_FUNCTION
  void operator() (int cell, int pt) const;

  void finalize(
      CP::StateMechanical<ScalarT, CP::MAX_DIM> const & state_mechanical,
      CP::StateInternal<ScalarT, CP::MAX_SLIP> const & state_internal,
      utility::StaticPointer<CP::Integrator<EvalT, CP::MAX_DIM, CP::MAX_SLIP>> const & integrator,
      int const cell,
      int const pt) const;

  ///
  ///  Set a NOX status test to Failed, which will trigger Piro to cut the global
  ///  load step, assuming the load-step-reduction feature is active.
  /// FIXME: This needs to be done outside of the material point loop
  /// (it's a race condition)
  void
  forceGlobalLoadStepReduction(std::string const & message) const
  {
    ALBANY_ASSERT(nox_status_test_.is_null() == false, "Invalid NOX status test");
    nox_status_test_->status_ = NOX::StatusTest::Failed;
    nox_status_test_->status_message_ = message;
  }

private:

  ///
  /// Crystal elasticity parameters
  ///
  RealType
  c11_{0.0};

  RealType
  c12_{0.0};

  RealType
  c13_{0.0};

  RealType
  c33_{0.0};

  RealType
  c44_{0.0};

  RealType
  c66_{0.0};

  RealType
  c11_temperature_coeff_{0.0};

  RealType
  c12_temperature_coeff_{0.0};

  RealType
  c13_temperature_coeff_{0.0};

  RealType
  c33_temperature_coeff_{0.0};

  RealType
  c44_temperature_coeff_{0.0};

  RealType
  c66_temperature_coeff_{0.0};

  RealType
  reference_temperature_{0.0};

  RealType
  norm_slip_residual_{0.0};

  int
  num_iter_residual_{0};

  minitensor::Tensor<RealType, CP::MAX_DIM>
  element_block_orientation_;

  /// Number of slip families
  int
  num_family_{0};

  /// Number of slip systems
  int
  num_slip_{0};

  // Index in global element numbering
  int
  index_element_{0};

  /// Unrotated elasticity tensor
  minitensor::Tensor4<ScalarT, CP::MAX_DIM>
  C_unrotated_;

  /// Vector of structs holding slip system family data
  std::vector<CP::SlipFamily<CP::MAX_DIM, CP::MAX_SLIP>>
  slip_families_;

  /// Vector of structs holding slip system data
  std::vector<CP::SlipSystem<CP::MAX_DIM>>
  slip_systems_;

  /// Flags for reading lattice orientations from file
  bool
  read_orientations_from_mesh_{false};

  ///
  /// Solution options
  ///
  CP::IntegrationScheme
  integration_scheme_{CP::IntegrationScheme::UNDEFINED};

  CP::ResidualType
  residual_type_{CP::ResidualType::UNDEFINED};

  CP::PredictorSlip
  predictor_slip_{CP::PredictorSlip::UNDEFINED};

  minitensor::StepType
  step_type_{minitensor::StepType::UNDEFINED};

  /// Minisolver Minimizer
  minitensor::Minimizer<ValueT, CP::NLS_DIM>
  minimizer_;

  /// ROL Minimizer
  ROL::MiniTensor_Minimizer<ValueT, CP::NLS_DIM>
  rol_minimizer_;

  ///
  /// Output options
  ///
  CP::Verbosity
  verbosity_{CP::Verbosity::UNDEFINED};

  bool
  write_data_file_{false};

  ///
  /// Dependent MDFields
  ///
  ConstScalarField
  def_grad_;

  ConstScalarField
  time_;

  ConstScalarField
  delta_time_;

  ConstScalarField
  temperature_;

  ///
  /// Evaluated MDFields
  ///
  ScalarField
  eqps_;

  ScalarField
  xtal_rotation_;

  ScalarField
  stress_;

  ScalarField
  plastic_deformation_;

  ScalarField
  velocity_gradient_;

  ScalarField
  velocity_gradient_plastic_;

  ScalarField
  cp_residual_;

  ScalarField
  cp_residual_iter_;

  ScalarField
  source_;

  std::vector<Teuchos::RCP<ScalarField>>
  slips_;

  std::vector<Albany::MDArray *>
  previous_slips_;

  std::vector<Teuchos::RCP<ScalarField>>
  slip_rates_;

  std::vector<Albany::MDArray *>
  previous_slip_rates_;

  std::vector<Teuchos::RCP<ScalarField>>
  hards_;

  std::vector<Albany::MDArray *>
  previous_hards_;

  std::vector<Teuchos::RCP<ScalarField>>
  shears_;

  //
  // Field strings
  //
  std::string const
  eqps_string_ = field_name_map_["eqps"];

  std::string const
  Re_string_ = field_name_map_["Re"];

  std::string const
  cauchy_string_ = field_name_map_["Cauchy_Stress"];

  std::string const
  Fp_string_ = field_name_map_["Fp"];

  std::string const
  L_string_ = field_name_map_["Velocity_Gradient"];

  std::string const
  Lp_string_ = field_name_map_["Velocity_Gradient_Plastic"];

  std::string const
  residual_string_ = field_name_map_["CP_Residual"];

  std::string const
  residual_iter_string_ = field_name_map_["CP_Residual_Iter"];

  std::string const
  source_string_ = field_name_map_["Mechanical_Source"];

  std::string const
  F_string_ = field_name_map_["F"];

  std::string const
  J_string_ = field_name_map_["J"];

  std::string const
  time_string_ = "Time";

  std::string const
  dt_string_ = "Delta Time";

  std::string const
  temperature_string_ = "Temperature";

  ///
  /// State Variables
  ///
  Albany::MDArray
  previous_plastic_deformation_;

  Albany::MDArray
  previous_defgrad_;

  RealType
  dt_{0.0};

  Teuchos::ArrayRCP<RealType*>
  rotation_matrix_transpose_;

};

template<typename EvalT, typename Traits>
class CrystalPlasticityModel : public LCM::ParallelConstitutiveModel<EvalT, Traits, CrystalPlasticityKernel<EvalT, Traits>> {
public:
  CrystalPlasticityModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);
};

} // namespace LCM

#endif
