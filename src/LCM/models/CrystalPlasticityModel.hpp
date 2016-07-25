//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_hpp)
#define LCM_CrystalPlasticityModel_hpp

#include "core/CrystalPlasticity/CrystalPlasticityCore.hpp"
#include "core/CrystalPlasticity/NonlinearSolver.hpp"
#include "ConstitutiveModel.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"

namespace LCM
{

//! \brief CrystalPlasticity Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class CrystalPlasticityModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  enum class IntegrationScheme
  {
    UNDEFINED = 0, 
    EXPLICIT = 1, 
    IMPLICIT = 2
  };

  enum class ResidualType
  {
    UNDEFINED = 0, 
    SLIP = 1, 
    SLIP_HARDNESS = 2
  };

  using ScalarT = typename EvalT::ScalarT;

  // Dimension of problem, e.g., 2 -> 2D, 3 -> 3D
  using ConstitutiveModel<EvalT, Traits>::num_dims_;

  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  // optional temperature support
  using ConstitutiveModel<EvalT, Traits>::have_temperature_;
  using ConstitutiveModel<EvalT, Traits>::expansion_coeff_;
  using ConstitutiveModel<EvalT, Traits>::ref_temperature_;
  using ConstitutiveModel<EvalT, Traits>::heat_capacity_;
  using ConstitutiveModel<EvalT, Traits>::density_;
  using ConstitutiveModel<EvalT, Traits>::temperature_;
  
  ///
  /// Constructor
  ///
  CrystalPlasticityModel(
      Teuchos::ParameterList* p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~CrystalPlasticityModel()
  {
  }

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

  virtual
  void
  computeStateParallel(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

  ///
  ///  Set a NOX status test to Failed, which will trigger Piro to cut the global
  ///  load step, assuming the load-step-reduction feature is active.
  ///
  void
  forceGlobalLoadStepReduction()
  {
    TEUCHOS_TEST_FOR_EXCEPTION(
        nox_status_test_.is_null(),
        std::logic_error,
        "\n**** Error in CrystalPlasticityModel: \
            error accessing NOX status test.");

    nox_status_test_->status_ = NOX::StatusTest::Failed;
  }

private:

  ///
  /// Private to prohibit copying
  ///
  CrystalPlasticityModel(const CrystalPlasticityModel &);

  ///
  /// Private to prohibit copying
  ///
  CrystalPlasticityModel & operator=(const CrystalPlasticityModel &);

  ///
  /// Crystal elasticity parameters
  ///
  RealType
  c11_;

  RealType
  c12_;

  RealType
  c44_;

  RealType
  c11_temperature_coeff_;

  RealType
  c12_temperature_coeff_;

  RealType
  c44_temperature_coeff_;

  RealType
  reference_temperature_;
  
  Intrepid2::Tensor<RealType, CP::MAX_DIM>
  element_block_orientation_;

  ///
  /// Number of slip systems
  ///
  int
  num_family_;

  int
  num_slip_;

  ///
  /// Unrotated elasticity tensor
  ///
  Intrepid2::Tensor4<ScalarT, CP::MAX_DIM>
  C_unrotated_;

  ///
  /// Elasticity tensor
  ///
  Intrepid2::Tensor4<ScalarT, CP::MAX_DIM>
  C_;

  //
  // Unrotated slip directions
  //
  std::vector<Intrepid2::Vector<RealType, CP::MAX_DIM>>
  s_unrotated_;

  //
  // Unrotated slip normals
  //
  std::vector<Intrepid2::Vector<RealType, CP::MAX_DIM>>
  n_unrotated_;

  ///
  /// Vector holding slip system families
  ///
  std::vector<CP::SlipFamily<CP::MAX_DIM, CP::MAX_SLIP>>
  slip_families_;

  ///
  /// Struct holding slip system data
  ///
  std::vector<CP::SlipSystem<CP::MAX_DIM>>
  slip_systems_;

  ///
  /// Flags for reading lattice orientations from file
  ///
  bool read_orientations_from_mesh_;

  ///
  /// Solution options
  ///
  IntegrationScheme 
  integration_scheme_;

  ResidualType
  residual_type_;

  bool
  apply_slip_predictor_;
  
  Intrepid2::StepType
  step_type_;

  RealType
  nonlinear_solver_relative_tolerance_;
  
  RealType
  nonlinear_solver_absolute_tolerance_;
  
  int
  nonlinear_solver_max_iterations_;
  
  int
  nonlinear_solver_min_iterations_;

  ///
  /// Pointer to NOX status test, allows the material model to force a global load step reduction
  ///
  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>
  nox_status_test_;

  ///
  /// Output options 
  ///
  int 
  verbosity_;

  bool
  write_data_file_;
};

}

#endif
