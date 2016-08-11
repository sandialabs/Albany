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
#include "../parallel_models/ParallelConstitutiveModel.hpp"
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
  using BaseKernel::temperature_;
  
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;
  using BaseKernel::addStateVariable;
  using BaseKernel::extractEvaluatedFieldArray;

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
  /// Virtual Denstructor
  ///
  virtual
  ~CrystalPlasticityKernel()
  {
  }

  void
  init(Workset & workset,
       FieldMap<ScalarT> & dep_fields,
       FieldMap<ScalarT> & eval_fields);

  ///
  /// Method to compute the state for a single cell and quadrature point
  //  (e.g. energy, stress, tangent)
  ///
  KOKKOS_INLINE_FUNCTION
  void operator() (int cell, int pt) const;

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
  /// Crystal elasticity parameters
  ///
  RealType
  c11_;

  RealType
  c12_;

  RealType
  c13_;

  RealType
  c33_;

  RealType
  c44_;

  RealType
  c66_;

  RealType
  c11_temperature_coeff_;

  RealType
  c12_temperature_coeff_;

  RealType
  c13_temperature_coeff_;

  RealType
  c33_temperature_coeff_;

  RealType
  c44_temperature_coeff_;

  RealType
  c66_temperature_coeff_;

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
  bool
  read_orientations_from_mesh_;

  ///
  /// Solution options
  ///
  CP::IntegrationScheme 
  integration_scheme_;

  CP::ResidualType
  residual_type_;

  bool
  apply_slip_predictor_;
  
  Intrepid2::StepType
  step_type_;

  ///
  /// Minimizer
  ///
  Intrepid2::Minimizer<ValueT, CP::NLS_DIM>
  minimizer_;

  ///
  /// Pointer to NOX status test, allows the material model to force a global load step reduction
  ///
  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>
  nox_status_test_;
  
  ///
  /// Memory management
  ///
  //mutable utility::StaticAllocator
  //allocator_;

  ///
  /// Output options 
  ///
  int 
  verbosity_;

  bool
  write_data_file_;

  ///
  /// Dependent MDFields
  ///

  ScalarField def_grad_;
  ScalarField delta_time_;

  ///
  /// Evaluated MDFields
  ///
  ScalarField eqps_;
  ScalarField xtal_rotation_;
  ScalarField stress_;
  ScalarField plastic_deformation_;
  ScalarField velocity_gradient_;
  ScalarField source_;
  ScalarField cp_residual_;
  ScalarField cp_residual_iter_;
  ScalarField time_;

  std::vector<Teuchos::RCP<ScalarField>> slips_;
  std::vector<Albany::MDArray *> previous_slips_;

  std::vector<Teuchos::RCP<ScalarField>> slip_rates_;
  std::vector<Albany::MDArray *> previous_slip_rates_;

  std::vector<Teuchos::RCP<ScalarField>> hards_;
  std::vector<Albany::MDArray *> previous_hards_;

  std::vector<Teuchos::RCP<ScalarField>> shears_;

  ///
  /// State Variables
  ///
  Albany::MDArray previous_plastic_deformation_;
  RealType dt_;

  Teuchos::ArrayRCP<double*>
  rotation_matrix_transpose_;


};

template<typename EvalT, typename Traits>
using CrystalPlasticityModel
  = LCM::ParallelConstitutiveModel<EvalT, Traits, CrystalPlasticityKernel<EvalT, Traits>>;

}

#endif
