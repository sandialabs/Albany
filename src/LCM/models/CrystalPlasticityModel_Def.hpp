//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
/*
#include "Teuchos_LAPACK.hpp"
#include <Tsqr_Matrix.hpp>
//*/
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

#include <MiniLinearSolver.h>

#include <typeinfo>
#include <iostream>
#include <Sacado_Traits.hpp>

#include "core/CrystalPlasticity/ParameterReader.hpp"
#include <type_traits>

namespace
{
// Matches ScalarT != ST
template<class T, typename std::enable_if< !std::is_same<T, ST>::value>::type* = nullptr >
bool isnaninf(const T& x)
{
  typedef typename Sacado::ValueType<T>::type ValueT;

  if (Teuchos::ScalarTraits<ValueT>::isnaninf(x.val())) {
    return true;
  }

  for (int i=0; i<x.size(); i++) {
    if (Teuchos::ScalarTraits<ValueT>::isnaninf(x.dx(i))) {
      return true;
    }
  }

  return false;
}

// Matches ScalarT == ST
template<class T, typename std::enable_if< std::is_same<T, ST>::value>::type* = nullptr >
bool
isnaninf(const T & x)
{
  return Teuchos::ScalarTraits<T>::isnaninf(x);
}

} // anonymous namespace

namespace LCM
{
template<typename EvalT, typename Traits>
CrystalPlasticityKernel<EvalT, Traits>::
CrystalPlasticityKernel(
    ConstitutiveModel<EvalT, Traits> & model,
    Teuchos::ParameterList* p,
    Teuchos::RCP<Albany::Layouts> const & dl)
  : BaseKernel(model),
    num_family_(p->get<int>("Number of Slip Families", 1)),
    num_slip_(p->get<int>("Number of Slip Systems", 0))
{
	CP::ParameterReader<EvalT, Traits>
  preader(p);

  slip_systems_.resize(num_slip_);

  // Store an RCP to the NOX status test, if available
  if (p->isParameter("NOX Status Test") == true) {
    nox_status_test_ =
        p->get<Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>>(
            "NOX Status Test");
  } else {
    nox_status_test_ = Teuchos::rcp(new NOX::StatusTest::ModelEvaluatorFlag);
    p->set<Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>>(
        "NOX Status Test", nox_status_test_);
  }

  Teuchos::ParameterList
  e_list = p->sublist("Crystal Elasticity");

  if (p->isParameter("Read Lattice Orientation From Mesh")) {
    read_orientations_from_mesh_ = true;
  }
  else {
    read_orientations_from_mesh_ = false;
    // TODO check if basis is given else default
    // NOTE default to coordinate axes; construct 3rd direction if only 2 given
    element_block_orientation_.set_dimension(CP::MAX_DIM);
    for (int i = 0; i < CP::MAX_DIM; ++i) {
      std::vector<RealType> const
      b_temp = e_list.get<Teuchos::Array<RealType>>(
        Albany::strint("Basis Vector", i + 1)).toVector();

      minitensor::Vector<RealType, CP::MAX_DIM>
      basis(CP::MAX_DIM);

      for (int dim = 0; dim < CP::MAX_DIM; ++dim){
        basis[dim] = b_temp[dim];
      }

      basis = minitensor::unit(basis);

      // TODO check zero, rh system
      // Filling columns of transformation with basis vectors
      // We are forming R^{T} which is equivalent to the direction cosine matrix
      for (int j = 0; j < CP::MAX_DIM; ++j) {
        element_block_orientation_(j, i) = basis[j];
      }
    }
  }

  verbosity_ = preader.getVerbosity();

	integration_scheme_ = preader.getIntegrationScheme();
  residual_type_ = preader.getResidualType();
	step_type_ = preader.getStepType();
	minimizer_ = preader.getMinimizer();
	rol_minimizer_ = preader.getRolMinimizer();
  predictor_slip_ = preader.getPredictorSlip();
						   
  // ensure minimizer abs tolerance isn't too low
  ALBANY_ASSERT(minimizer_.abs_tol >= CP::MIN_TOL,
		"Specified absolute tolerance is too tight:"
		" minimum tolerance: 1.0e-14");

  if (verbosity_ >= CP::Verbosity::HIGH) {
    std::cout << "Slip predictor: " << int(predictor_slip_) << std::endl;
  }

  write_data_file_ = p->get<bool>("Write Data File", false);

  if (verbosity_ >= CP::Verbosity::HIGH) {
    std::cout << ">>> in cp constructor\n";
    std::cout << ">>> parameter list:\n" << *p << std::endl;
  }

  //
  // Obtain crystal elasticity constants and populate elasticity tensor
  // assuming cubic symmetry (fcc, bcc) or transverse isotropy (hcp)
  // Constants C11, C12, and C44 must be defined for either symmetry
  c11_ = e_list.get<RealType>("C11");
  c12_ = e_list.get<RealType>("C12");
  c13_ = e_list.get<RealType>("C13", c12_);
  c33_ = e_list.get<RealType>("C33", c11_);
  c44_ = e_list.get<RealType>("C44");
  c11_temperature_coeff_ = e_list.get<RealType>("M11", NAN);
  c12_temperature_coeff_ = e_list.get<RealType>("M12", NAN);
  c13_temperature_coeff_ = e_list.get<RealType>("M13", c12_temperature_coeff_);
  c33_temperature_coeff_ = e_list.get<RealType>("M33", c11_temperature_coeff_);
  c44_temperature_coeff_ = e_list.get<RealType>("M44", NAN);
  reference_temperature_ = e_list.get<RealType>("Reference Temperature", NAN);

  C_unrotated_.set_dimension(CP::MAX_DIM);
  if (c11_ == c33_) {
    c66_ = c44_;
    c66_temperature_coeff_ = c44_temperature_coeff_;
  } else {
    c66_ = 0.5 * (c11_ - c12_);
    c66_temperature_coeff_ =
      0.5 * (c11_temperature_coeff_ - c12_temperature_coeff_);
  }

  CP::computeElasticityTensor(c11_, c12_, c13_, c33_, c44_, c66_, C_unrotated_);

  if (verbosity_ >= CP::Verbosity::HIGH) {
    // print elasticity tensor
    std::cout << ">>> Unrotated C :" << std::endl << C_unrotated_ << std::endl;
  }


  //
  // Get slip families
  //
  slip_families_.reserve(num_family_);
  for (int num_fam(0); num_fam < num_family_; ++num_fam) {
    slip_families_.emplace_back(preader.getSlipFamily(num_fam));
  }

  //
  // Get slip system information
  //
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    Teuchos::ParameterList
    ss_list = p->sublist(Albany::strint("Slip System", num_ss + 1));

    CP::SlipSystem<CP::MAX_DIM> &
    slip_system = slip_systems_.at(num_ss);

    slip_system.slip_family_index_ = ss_list.get<int>("Slip Family", 0);

    CP::SlipFamily<CP::MAX_DIM, CP::MAX_SLIP> &
    slip_family = slip_families_[slip_system.slip_family_index_];

    minitensor::Index
    slip_system_index = slip_family.num_slip_sys_;

    slip_family.slip_system_indices_[slip_system_index] = num_ss;

    slip_family.num_slip_sys_++;

    //
    // Read and normalize slip directions. Miller indices need to be normalized.
    //
    std::vector<RealType>
    s_temp = ss_list.get<Teuchos::Array<RealType>>("Slip Direction").toVector();

    minitensor::Vector<RealType, CP::MAX_DIM>
    s_temp_normalized(CP::MAX_DIM);

    for (int i = 0; i < CP::MAX_DIM; ++i) {
      s_temp_normalized[i] = s_temp[i];
    }
    s_temp_normalized = minitensor::unit(s_temp_normalized);
    slip_systems_.at(num_ss).s_.set_dimension(CP::MAX_DIM);
    slip_systems_.at(num_ss).s_ = s_temp_normalized;

    //
    // Read and normalize slip normals. Miller indices need to be normalized.
    //
    std::vector<RealType>
    n_temp = ss_list.get<Teuchos::Array<RealType>>("Slip Normal").toVector();

    minitensor::Vector<RealType, CP::MAX_DIM>
    n_temp_normalized(CP::MAX_DIM);

    for (int i = 0; i < CP::MAX_DIM; ++i) {
      n_temp_normalized[i] = n_temp[i];
    }

    n_temp_normalized = minitensor::unit(n_temp_normalized);
    slip_systems_.at(num_ss).n_.set_dimension(CP::MAX_DIM);
    slip_systems_.at(num_ss).n_ = n_temp_normalized;

    slip_systems_.at(num_ss).projector_.set_dimension(CP::MAX_DIM);
    slip_systems_.at(num_ss).projector_ =
      minitensor::dyad(slip_systems_.at(num_ss).s_, slip_systems_.at(num_ss).n_);

    auto const
    index_param =
      slip_family.phardening_parameters_->param_map_["Initial Hardening State"];

    RealType const
    state_hardening_initial =
      slip_family.phardening_parameters_->getParameter(index_param);

    slip_system.state_hardening_initial_ =
      ss_list.get<RealType>("Initial Hardening State", state_hardening_initial);
  }

  for (int sf_index(0); sf_index < num_family_; ++sf_index)
  {
    auto &
    slip_family = slip_families_[sf_index];

    // Set the saturated hardness value, if applicable
    slip_family.phardening_parameters_->setValueAsymptotic();

    // Create latent matrix for hardening law
    slip_family.phardening_parameters_->createLatentMatrix(
      slip_family, slip_systems_);

    if (verbosity_ >= CP::Verbosity::HIGH) {
      std::cout << slip_family.latent_matrix_ << std::endl;
    }

    slip_family.slip_system_indices_.set_dimension(slip_family.num_slip_sys_);

    if (verbosity_ >= CP::Verbosity::HIGH) {
      std::cout << "slip system indices";
      std::cout << slip_family.slip_system_indices_ << std::endl;
    }
  }

  //
  // Define the dependent fields required for calculation
  //
  setDependentField(F_string_, dl->qp_tensor);
  setDependentField(J_string_, dl->qp_scalar);
  setDependentField(dt_string_, dl->workset_scalar);
  if (write_data_file_) {
    setDependentField(time_string_, dl->workset_scalar);
  }

  //
  // Define the evaluated fields
  //
  setEvaluatedField(eqps_string_, dl->qp_scalar);
  setEvaluatedField(Re_string_, dl->qp_tensor);
  setEvaluatedField(cauchy_string_, dl->qp_tensor);
  setEvaluatedField(Fp_string_, dl->qp_tensor);
  setEvaluatedField(L_string_, dl->qp_tensor);
  setEvaluatedField(Lp_string_, dl->qp_tensor);
  setEvaluatedField(residual_string_, dl->qp_scalar);
  setEvaluatedField(residual_iter_string_, dl->qp_scalar);

  if (have_temperature_) {
    setDependentField(temperature_string_, dl->qp_scalar);
    setEvaluatedField(source_string_, dl->qp_scalar);
  }

  //
  // define the state variables
  //

  // eqps
  addStateVariable(eqps_string_, dl->qp_scalar, "scalar", 0.0, false,
      p->get<bool>("Output eqps", false));

  // Re
  addStateVariable(Re_string_, dl->qp_tensor, "identity", 0.0, false,
      p->get<bool>("Output Re", false));

  // stress
  addStateVariable(cauchy_string_, dl->qp_tensor, "scalar", 0.0, false,
      p->get<bool>("Output Cauchy Stress", false));

  // Fp
  addStateVariable(Fp_string_, dl->qp_tensor, "identity", 0.0, true,
      p->get<bool>("Output Fp", false));

  // L
  addStateVariable(L_string_, dl->qp_tensor, "scalar", 0.0, false,
      p->get<bool>("Output L", false));

  // Lp
  addStateVariable(Lp_string_, dl->qp_tensor, "scalar", 0.0, false,
      p->get<bool>("Output Lp", false));

  // mechanical source
  if (have_temperature_) {
    addStateVariable(source_string_, dl->qp_scalar, "scalar", 0.0, false,
        p->get<bool>("Output Mechanical Source", false));
  }

  // gammas for each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    g = Albany::strint("gamma", num_ss + 1, '_');

    std::string const
    gamma_string = field_name_map_[g];

    std::string const
    output_gamma_string = "Output " + gamma_string;

    setEvaluatedField(gamma_string, dl->qp_scalar);
    addStateVariable(gamma_string, dl->qp_scalar, "scalar", 0.0, true,
        p->get<bool>(output_gamma_string, false));
  }

  // gammadots for each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    g_dot = Albany::strint("gamma_dot", num_ss + 1, '_');

    std::string const
    gamma_dot_string = field_name_map_[g_dot];

    setEvaluatedField(gamma_dot_string, dl->qp_scalar);

    std::string const
    output_gamma_dot_string = "Output " + gamma_dot_string;

    addStateVariable(gamma_dot_string, dl->qp_scalar, "scalar", 0.0, true,
        p->get<bool>(output_gamma_dot_string, false));
  }

  // tau_hard - state variable for hardening on each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    t_h = Albany::strint("tau_hard", num_ss + 1, '_');

    std::string const
    tau_hard_string = field_name_map_[t_h];

    auto const initial = slip_systems_.at(num_ss).state_hardening_initial_;

    setEvaluatedField(tau_hard_string, dl->qp_scalar);

    std::string const
    output_tau_hard_string = "Output " + tau_hard_string;

    addStateVariable(tau_hard_string, dl->qp_scalar, "scalar", initial, true,
        p->get<bool>(output_tau_hard_string, false));
  }

  // taus - output resolved shear stress for debugging - not stated
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    t = Albany::strint("tau", num_ss + 1, '_');

    std::string const
    tau_string = field_name_map_[t];

    setEvaluatedField(tau_string, dl->qp_scalar);

    std::string const
    output_tau_string = "Output " + tau_string;

    addStateVariable(tau_string, dl->qp_scalar, "scalar", 0.0, false,
        p->get<bool>(output_tau_string, false));
  }

  // residual
  addStateVariable(residual_string_, dl->qp_scalar, "scalar", 0.0, false,
      p->get<bool>("Output CP_Residual", false));

  // residual iterations
  addStateVariable(residual_iter_string_, dl->qp_scalar, "scalar", 0.0, false,
      p->get<bool>("Output CP_Residual_Iter", false));
}


//
// Initialize state for computing the constitutive response of the material
//
template<typename EvalT, typename Traits>
void CrystalPlasticityKernel<EvalT, Traits>::init(
    Workset & workset,
    FieldMap<const ScalarT> & dep_fields,
    FieldMap<ScalarT> & eval_fields)
{
  if(verbosity_ == CP::Verbosity::EXTREME) {
    index_element_ = workset.wsIndex;
  }
  else {
    index_element_ = -1;
  }

  if(verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << ">>> kernel::init\n";
  }

  if (read_orientations_from_mesh_)
  {
    rotation_matrix_transpose_ = workset.wsLatticeOrientation;
    ALBANY_ASSERT(rotation_matrix_transpose_.is_null() == false,
        "Rotation matrix not found on genesis mesh");
  }

  //
  // extract dependent MDFields
  //
  def_grad_ = *dep_fields[F_string_];
  if (write_data_file_) {
    time_ = *dep_fields[time_string_];
  }
  delta_time_ = *dep_fields[dt_string_];
  if (have_temperature_) {
    temperature_ = *dep_fields[temperature_string_];
    source_ = *eval_fields[source_string_];
  }

  //
  // extract evaluated MDFields
  //
  eqps_ = *eval_fields[eqps_string_];
  xtal_rotation_ = *eval_fields[Re_string_];
  stress_ = *eval_fields[cauchy_string_];
  plastic_deformation_ = *eval_fields[Fp_string_];
  velocity_gradient_ = *eval_fields[L_string_];
  velocity_gradient_plastic_ = *eval_fields[Lp_string_];
  cp_residual_ = *eval_fields[residual_string_];
  cp_residual_iter_ = *eval_fields[residual_iter_string_];

  // extract slip on each slip system
  extractEvaluatedFieldArray("gamma", num_slip_, slips_, previous_slips_,
      eval_fields, workset);

  // extract slip rate on each slip system
  extractEvaluatedFieldArray("gamma_dot", num_slip_, slip_rates_,
    previous_slip_rates_, eval_fields, workset);

  // extract hardening on each slip system
  extractEvaluatedFieldArray("tau_hard", num_slip_, hards_, previous_hards_,
      eval_fields, workset);

  // store shear on each slip system for output
  extractEvaluatedFieldArray("tau", num_slip_, shears_, eval_fields);

  // get state variables

  previous_plastic_deformation_ = (*workset.stateArrayPtr)[Fp_string_ + "_old"];
  previous_defgrad_ = (*workset.stateArrayPtr)[F_string_ + "_old"];

  dt_ = SSV::eval(delta_time_(0));

  // Resest status and status message for model failure test
  //nox_status_test_->status_message_ = "";
  //nox_status_test_->status_ = NOX::StatusTest::Unevaluated;
}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
CrystalPlasticityKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  if(verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << ">>> in kernel::operator\n";
    std::cout << "    cell: " << cell << " point: " << pt << "\n";
  }
  // If a previous constitutive calculation has failed, exit immediately.
  if (nox_status_test_->status_ == NOX::StatusTest::Failed) {
    if (verbosity_ == CP::Verbosity::DEBUG) {
      std::cout << "  ****Returning on failed****" << std::endl;
    }
    return;
  }
  // TODO: In the future for CUDA this should be moved out of the kernel because
  // it uses dynamic allocation for the buffer. It should also be modified to use
  // cudaMalloc.
  utility::StaticAllocator
  allocator(1024 * 1024);

  //
  // Known quantities
  //
  minitensor::Tensor<RealType, CP::MAX_DIM>
  Fp_n(num_dims_);

  minitensor::Vector<RealType, CP::MAX_SLIP>
  slip_n(num_slip_);

  minitensor::Vector<RealType, CP::MAX_SLIP>
  slip_dot_n(num_slip_);

  minitensor::Vector<RealType, CP::MAX_SLIP>
  state_hardening_n(num_slip_);

  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  F_np1(num_dims_);

  minitensor::Tensor<RealType, CP::MAX_DIM>
  F_n(num_dims_);

  //
  // Unknown quantities
  //
  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  Lp_np1(num_dims_);

  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  Fp_np1(num_dims_);

  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  sigma_np1(num_dims_);

  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  S_np1(num_dims_);

  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  slip_np1(num_slip_);

  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  shear_np1(num_slip_);

  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  state_hardening_np1(num_slip_);

  ///
  /// Elasticity tensor
  ///
  minitensor::Tensor4<ScalarT, CP::MAX_DIM>
  C_unrotated = C_unrotated_;

  minitensor::Tensor4<ScalarT, CP::MAX_DIM>
  C(CP::MAX_DIM);

  RealType
  norm_slip_residual;

  minitensor::Tensor<RealType, CP::MAX_DIM>
  orientation_matrix(CP::MAX_DIM);

  std::vector<CP::SlipSystem<CP::MAX_DIM>>
  element_slip_systems = slip_systems_;

  if (have_temperature_)
  {
    RealType const
    tlocal = SSV::eval(temperature_(cell,pt));

    RealType const
    delta_temperature = tlocal - reference_temperature_;

    RealType const
    c11 = c11_ + c11_temperature_coeff_ * delta_temperature;

    RealType const
    c12 = c12_ + c12_temperature_coeff_ * delta_temperature;

    RealType const
    c13 = c13_ + c13_temperature_coeff_ * delta_temperature;

    RealType const
    c33 = c33_ + c33_temperature_coeff_ * delta_temperature;

    RealType const
    c44 = c44_ + c44_temperature_coeff_ * delta_temperature;

    RealType const
    c66 = c66_ + c66_temperature_coeff_ * delta_temperature;

    CP::computeElasticityTensor(c11, c12, c13, c33, c44, c66, C_unrotated);

    if (verbosity_ >= CP::Verbosity::HIGH) {
      std::cout << "tlocal: " << tlocal << std::endl;
      std::cout << "c11, c12, c44: " << c11 << c12 << c44 << std::endl;
    }
  }

  if (read_orientations_from_mesh_) {
    for (int i = 0; i < CP::MAX_DIM; ++i) {
      for (int j = 0; j < CP::MAX_DIM; ++j) {
        orientation_matrix(i,j) = rotation_matrix_transpose_[cell][i * CP::MAX_DIM + j];
      }
    }
  }
  else {
    orientation_matrix = element_block_orientation_;
  }

  // Set the rotated elasticity tensor, slip normals, slip directions,
  // and projection operator
  C = minitensor::kronecker(orientation_matrix, C_unrotated);
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    auto &
    slip_system = element_slip_systems.at(num_ss);

    slip_system.s_ = orientation_matrix * slip_systems_.at(num_ss).s_;
    slip_system.n_ = orientation_matrix * slip_systems_.at(num_ss).n_;
    slip_system.projector_ = minitensor::dyad(slip_system.s_, slip_system.n_);
  }

  // Copy data from Albany fields into local data structures
  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      F_np1(i, j) = def_grad_(cell, pt, i, j);
      Fp_n(i, j) = previous_plastic_deformation_(cell, pt, i, j);
      F_n(i, j) = previous_defgrad_(cell, pt, i, j);
    }
  }

  if (verbosity_ == CP::Verbosity::DEBUG)
  {
    std::cout << "F_n" << std::endl;
    std::cout << std::setprecision(4) << F_n << std::endl;
    std::cout << "F_np1" << std::endl;
    std::cout << std::setprecision(4) << F_np1 << std::endl;
  }

  // Bring in internal state from previous step
  for (int s(0); s < num_slip_; ++s) {
    slip_n[s] = (*(previous_slips_[s]))(cell, pt);
    slip_np1[s] = slip_n[s];
    slip_dot_n[s] = (*(previous_slip_rates_[s]))(cell, pt);
    state_hardening_n[s] = (*(previous_hards_[s]))(cell, pt);
    state_hardening_np1[s] = state_hardening_n[s];
  }

  //
  // Set up slip predictor to assign isochoric part of F_increment to Fp_increment
  //
  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  slip_resistance(num_slip_, minitensor::Filler::ZEROS);

  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  rates_slip(num_slip_, minitensor::Filler::ZEROS);

  if (dt_ > 0.0)
  {
    bool
    failed{false};
    switch (predictor_slip_)
    {
      case CP::PredictorSlip::NONE:
      {
      } break;

      case CP::PredictorSlip::RATE:
      {

        for (int s = 0; s < num_slip_; ++s)
        {
          rates_slip[s] = slip_dot_n[s];
          slip_np1[s] = slip_n[s] + dt_ * rates_slip[s];
        }

        if (verbosity_ == CP::Verbosity::DEBUG) {
          std::cout << slip_np1 <<std::endl;
        }

        CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
            slip_systems_,
            slip_families_,
            dt_,
            rates_slip,
            state_hardening_n,
            state_hardening_np1,
            slip_resistance,
            failed);

      } break;

      case CP::PredictorSlip::SOLVE:
      {
        minitensor::Tensor<RealType, CP::MAX_DIM> const
        inv_F = minitensor::inverse(F_n);

        minitensor::Tensor<RealType, CP::MAX_DIM> const
        eye = minitensor::identity<RealType, CP::MAX_DIM>(num_dims_);

        minitensor::Tensor<RealType, CP::MAX_DIM> const
        F_np1_peeled = LCM::peel_tensor<EvalT, RealType, CP::MAX_DIM, CP::MAX_DIM>()(F_np1);

        minitensor::Tensor<RealType, CP::MAX_DIM> const
        L = 1.0 / dt_ * (F_np1_peeled * inv_F - eye);

        if (minitensor::norm(L) < CP::MACHINE_EPS) {
          break;
        }

        auto const
        size_problem = std::max(num_slip_, num_dims_ * num_dims_);

        minitensor::Tensor<RealType, CP::MAX_SLIP>
        dyad_matrix(size_problem);

        dyad_matrix.fill(minitensor::Filler::ZEROS);

        for (int s = 0; s < num_slip_; ++s) {
          for (int d(0); d < CP::MAX_DIM * CP::MAX_DIM; ++d) {
            dyad_matrix(d, s) = element_slip_systems.at(s).projector_[d];
          }
        }

        minitensor::Tensor<RealType, CP::MAX_SLIP>
        U_svd(size_problem);
        minitensor::Tensor<RealType, CP::MAX_SLIP>
        S_svd(size_problem);
        minitensor::Tensor<RealType, CP::MAX_SLIP>
        V_svd(size_problem);

        boost::tie(U_svd, S_svd, V_svd) = minitensor::svd(dyad_matrix);

        for (int s(0); s < num_slip_; ++s) {
          S_svd(s, s) = S_svd(s, s) > 1.0e-12 ? 1.0 / S_svd(s,s) : 0.0;
        }

        minitensor::Tensor<RealType, CP::MAX_SLIP> const
        Pinv = V_svd * S_svd * S_svd * minitensor::transpose(V_svd);

        minitensor::Vector<RealType, CP::MAX_SLIP>
        L_vec(size_problem, minitensor::Filler::ZEROS);

        int const
        num_p = 100;

        RealType const
        inc_portion = 1.0 / num_p;

        RealType
        min_diff = CP::HUGE_;

        minitensor::Vector<RealType, CP::MAX_SLIP>
        rates_slip_trial(num_slip_, minitensor::Filler::ZEROS);

        minitensor::Vector<RealType, CP::MAX_SLIP>
        slip_np1_trial(num_slip_, minitensor::Filler::ZEROS);

        minitensor::Vector<RealType, CP::MAX_SLIP>
        hardening_np1_trial(num_slip_, minitensor::Filler::ZEROS);

        minitensor::Vector<RealType, CP::MAX_SLIP>
        slip_resistance_trial(num_slip_, minitensor::Filler::ZEROS);

        for (int p = 1; p < num_p; ++p)
        {
          RealType const
          portion_L = p * inc_portion;

          for (int i = 0; i < num_dims_; ++i) {
            for (int j = 0; j < num_dims_; ++j) {
              L_vec(i * num_dims_ + j) = portion_L * SSV::eval(L(i, j));
            }
          }

          minitensor::Vector<RealType, CP::MAX_SLIP> const
          dm_lv = minitensor::transpose(dyad_matrix) * L_vec;

          minitensor::Vector<RealType, CP::MAX_SLIP>
          rates_slip_trial = Pinv * dm_lv;

          RealType const
          limit_rate = 1e-8 * minitensor::norm(rates_slip_trial);

          for (int s(0); s < num_slip_; ++s) {
            rates_slip_trial[s] =
                std::abs(rates_slip_trial[s]) > limit_rate ? rates_slip_trial[s] : 0.0;
            slip_np1_trial[s] = slip_n[s] + dt_ * rates_slip_trial[s];
          }

          if (verbosity_ == CP::Verbosity::DEBUG)
          {
            std::cout << "P^T * L" << std::endl;
            std::cout << std::setprecision(4) << dm_lv << std::endl;

            std::cout << "Trial slip rates" << std::endl;
            std::cout << std::setprecision(4) << rates_slip_trial << std::endl;

            std::cout << "F_n" << std::endl;
            std::cout << std::setprecision(4) << F_n << std::endl;
            std::cout << "F_np1" << std::endl;
            std::cout << std::setprecision(4) << F_np1 << std::endl;
            std::cout << "dF" << std::endl;
            std::cout << std::setprecision(4) << F_np1 * inv_F << std::endl;
            std::cout << "L_vec" << std::endl;
            std::cout << std::setprecision(4) << L_vec << std::endl;
            std::cout << "Pinv" << std::endl;
            std::cout << std::setprecision(4) << Pinv << std::endl;
            std::cout << "exp(L * dt_) * F_n" << std::endl;
            std::cout << std::setprecision(4) << minitensor::exp(dt_ * L) * F_n << std::endl;
          }

          minitensor::Tensor<RealType, CP::MAX_DIM>
          Lp_trial(num_dims_, minitensor::Filler::ZEROS);

          minitensor::Vector<RealType, CP::MAX_SLIP>
          Lp_vec = dyad_matrix * rates_slip_trial;

          for (int i = 0; i < num_dims_; ++i) {
            for (int j = 0; j < num_dims_; ++j) {
              Lp_trial(i, j) = Lp_vec(i * num_dims_ + j);
            }
          }

          if (verbosity_ == CP::Verbosity::DEBUG)
          {
            std::cout << "L" << std::endl;
            std::cout << std::setprecision(4) << L << std::endl;

            std::cout << "Lp_trial" << std::endl;
            std::cout << std::setprecision(4) << Lp_trial << std::endl;

            std::cout << "exp(Lp * dt_)" << std::endl;
            std::cout << std::setprecision(4) << minitensor::exp(dt_ * Lp_trial)<< std::endl;
          }

          minitensor::Tensor<RealType, CP::MAX_DIM>
          Fp_np1_trial(num_dims_, minitensor::Filler::ZEROS);

          // Compute Lp_trial, and Fp_np1_trial
          CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP, RealType>(
              element_slip_systems,
              dt_,
              slip_n,
              slip_np1_trial,
              Fp_n,
              Lp_trial,
              Fp_np1_trial);
         
          if (verbosity_ == CP::Verbosity::DEBUG)
          {
            std::cout << "Lp_trial" << std::endl;
            std::cout << std::setprecision(4) << Lp_trial << std::endl;
          }

          // minitensor::Vector<RealType, CP::MAX_SLIP>
          // rates_hardening(num_slip_, minitensor::Filler::ZEROS);

          CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, RealType>(
            slip_systems_,
            slip_families_,
            dt_,
            rates_slip_trial,
            state_hardening_n,
            hardening_np1_trial,
            slip_resistance_trial,
            failed);

          minitensor::Vector<RealType, CP::MAX_SLIP>
          shear_np1_trial_2(num_slip_);

          for (int s{0}; s < num_slip_; ++s) {

            auto const
            slip_family = slip_families_[element_slip_systems.at(s).slip_family_index_];

            // using Params = SaturationHardeningParameters<NumDimT, NumSlipT>;
            // auto const
            // phardening_parameters = slip_family.phardening_parameters_;
            // auto const
            // driver_hardening = 2.0 * slip_family.latent_matrix_ * std::abs(rates_slip_trial);
            // auto const
            // rate_slip_reference = phardening_params->getParameter(Params::RATE_SLIP_REFERENCE);
            // auto const
            // exponent_saturation = phardening_params->getParameter(Params::EXPONENT_SATURATION);
            // auto const
            // rate_effective = minitensor::norm_1(rates_slip_trial);
            // auto const
            // ratio_rate = rate_effective / rate_slip_reference;
            // auto const
            // stress_saturation_initial = phardening_params->getParameter(Params::STRESS_SATURATION_INITIAL);
            // auto const
            // stress_saturation = stress_saturation_initial * std::pow(ratio_rate, exponent_saturation);
            // auto const
            // resistance_slip_initial = phardening_params->getParameter(Params::STATE_HARDENING_INITIAL);
            // auto const
            // diff_hardening = stress_saturation - resistance_slip_initial;
            // auto const
            // numerator = state_hardening_n * diff_hardening + stress_saturation * dt_ * rate_effective;
            // auto const
            // denominator = diff_hardening + dt_ * rate_effective; 

            // state_hardening_np1[ss_index_global] = numerator / denominator;

            using Params = CP::PowerLawFlowParameters;
            auto const
            pflow_parameters = slip_family.pflow_parameters_;
            RealType const
            m = pflow_parameters->getParameter(Params::EXPONENT_RATE);
            RealType const
            g0 = pflow_parameters->getParameter(Params::RATE_SLIP_REFERENCE);
            RealType const
            ratio_rate = rates_slip_trial[s] / g0;
            RealType const
            sign_rate = ratio_rate > 0.0 ? 1.0 : -1.0;
            shear_np1_trial_2[s] = hardening_np1_trial[s] * std::pow(std::abs(ratio_rate), 1.0 / m) * sign_rate;
          }

          bool
          failed{false};

          minitensor::Tensor4<RealType, CP::MAX_DIM> const
          C_peeled = LCM::peel_tensor4<EvalT, RealType, CP::MAX_DIM, CP::MAX_DIM>()(C);

          minitensor::Tensor<RealType, CP::MAX_DIM>
          sigma_np1(num_dims_);

          minitensor::Tensor<RealType, CP::MAX_DIM>
          S_np1(num_dims_);

          minitensor::Vector<RealType, CP::MAX_SLIP>
          shear_np1_trial(num_slip_);

          CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, RealType>(
              element_slip_systems,
              C_peeled,
              F_np1_peeled,
              Fp_np1_trial,
              sigma_np1,
              S_np1,
              shear_np1_trial,
              failed);
         
          if (verbosity_ == CP::Verbosity::DEBUG)
          {
            std::cout << "Fp_np1_trial" << std::endl;
            std::cout << std::setprecision(4) << Fp_np1_trial << std::endl;
            std::cout << "shear_np1_trial" << std::endl;
            std::cout << std::setprecision(4) << shear_np1_trial << std::endl;
            std::cout << "shear_np1_trial_2" << std::endl;
            std::cout << std::setprecision(4) << shear_np1_trial_2 << std::endl;
          }

          // Ensure that the stress was calculated properly
          if (failed == true) {
            forceGlobalLoadStepReduction("Failed on initial guess");
            return;
          }

          minitensor::Tensor<RealType, CP::MAX_DIM> const
          F_e = F_np1_peeled * minitensor::inverse(Fp_np1_trial);

          minitensor::Tensor<RealType, CP::MAX_DIM> const
          C_e = minitensor::transpose(F_e) * F_e;

          minitensor::Tensor<RealType, CP::MAX_DIM> const
          stress_intermediate = C_e * S_np1;

          RealType const
          power_plastic = minitensor::dotdot(Lp_trial, stress_intermediate);
         
          if (verbosity_ == CP::Verbosity::DEBUG)
          {
            std::cout << "hardening_np1_trial" << std::endl;
            std::cout << std::setprecision(4) << hardening_np1_trial << std::endl;
          }

          if (failed) {
            this->forceGlobalLoadStepReduction("Failed on hardness");
            return;
          }

          minitensor::Vector<RealType, CP::MAX_SLIP>
          correction_hardening(num_slip_, minitensor::Filler::ONES);

          // for (int s(0); s < num_slip_; ++s) {
          //   correction_hardening[s] = 1.0 - 1.0 / hardening_np1_trial[s];
          // }

          RealType
          dissipation{0.0};

          for (int s(0); s < num_slip_; ++s) {
            dissipation += 
                shear_np1_trial[s] * rates_slip_trial[s] * correction_hardening[s];
          }

          RealType const
          diff = std::abs(minitensor::dot(rates_slip_trial, shear_np1_trial) - 
              minitensor::dot(rates_slip_trial, shear_np1_trial_2));

          if (diff < min_diff) {
            min_diff = diff;
            for (int s(0); s < num_slip_; ++s) {
              slip_np1[s] = slip_n[s] + dt_ * rates_slip_trial[s];
              state_hardening_np1[s] = hardening_np1_trial[s];
              slip_resistance[s] = slip_resistance_trial[s];
            }
          }

          if (verbosity_ == CP::Verbosity::DEBUG) {
            std::cout << "Plastic power" <<std::endl;
            std::cout << minitensor::dot(rates_slip_trial, shear_np1_trial) << std::endl;
            std::cout << minitensor::dot(rates_slip_trial, shear_np1_trial_2) << std::endl;
            // std::cout << "slip_np1 " << &slip_np1 << std::endl;
            // std::cout << slip_np1 <<std::endl;
          }
        }
      } break;

      default:
      {
      } break;
    }
  }

  if(verbosity_ == CP::Verbosity::DEBUG)
  {
    for (int s(0); s < num_slip_; ++s) {
      std::cout << "Slip on system " << s << " before predictor: ";
      std::cout << slip_n[s] << std::endl;
    }
    for (int s(0); s < num_slip_; ++s) {
      std::cout << "Slip rate on system " << s << " is: ";
      std::cout << slip_dot_n[s] << std::endl;
    }
    for (int s(0); s < num_slip_; ++s) {
      std::cout << "Slip on system " << s << " after predictor: ";
      std::cout << slip_np1[s] << std::endl;
    }
  }

  CP::StateMechanical<ScalarT, CP::MAX_DIM>
  state_mechanical(num_dims_, F_n, Fp_n, F_np1);

  CP::StateInternal<ScalarT, CP::MAX_SLIP>
  state_internal(index_element_, pt, num_slip_, state_hardening_n, slip_n);

  for (int s(0); s < num_slip_; ++s) {
    state_internal.rates_slip_[s] = rates_slip[s];
    state_internal.slip_np1_[s] = slip_np1[s];
    state_internal.hardening_np1_[s] = state_hardening_np1[s];
    state_internal.resistance_[s] = slip_resistance[s];
  }
  
  if (dt_ == 0.0)
  {
    if (verbosity_ == CP::Verbosity::EXTREME)
    {
      std::ofstream
      outfile;

      std::stringstream
      ss;

      ss << "slips_" << index_element_ << "_" << pt <<  ".out";

      std::string
      file = ss.str();

      outfile.open(file);
      outfile.close();
    }
        
  }

  auto
  integratorFactory = CP::IntegratorFactory<EvalT, CP::MAX_DIM, CP::MAX_SLIP>(
    allocator,
    minimizer_,
    rol_minimizer_,
    step_type_,
    nox_status_test_,
    element_slip_systems,
    slip_families_,
    state_mechanical,
    state_internal,
    C,
    dt_,
    verbosity_);

  utility::StaticPointer<CP::Integrator<EvalT, CP::MAX_DIM, CP::MAX_SLIP>>
  integrator = integratorFactory(integration_scheme_, residual_type_);

  integrator->update();

  if (verbosity_ >= CP::Verbosity::MEDIUM) {
    std::cout << "Fp_{n+1}" << std::endl;
    std::cout << state_mechanical.Fp_np1_ << std::endl;
    std::cout << "sigma_{n+1}" << std::endl;
    std::cout << state_mechanical.sigma_np1_ << std::endl;
    std::cout << "g_{n+1}" << std::endl;
    std::cout << state_internal.hardening_np1_ << std::endl;
  }

  // Check to make sure there is only one status test
  ALBANY_ASSERT(integrator->getStatus() == nox_status_test_->status_);

  // Exit early if update state is not successful
  if(nox_status_test_->status_ == NOX::StatusTest::Failed) {
    return;
  }

  finalize(
    state_mechanical,
    state_internal,
    integrator,
    cell,
    pt);

  if(write_data_file_) {
    if (cell == 0 && pt == 0)
    {
      std::ofstream
      data_file("output.dat", std::fstream::app);

      minitensor::Tensor<RealType, CP::MAX_DIM>
      P(num_dims_);

      data_file << "\n" << "time: ";
      data_file << std::setprecision(12);
      data_file << SSV::eval(time_(0));
      data_file << "     dt: ";
      data_file << std::setprecision(12) << dt_ << " \n";

      for (int s(0); s < num_slip_; ++s) {
        data_file << "\n" << "P" << s << ": ";
        P = element_slip_systems.at(s).projector_;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            data_file << std::setprecision(12);
            data_file << SSV::eval(P(i,j)) << " ";
          }
        }
      }

      for (int s(0); s < num_slip_; ++s) {
        data_file << "\n" << "slips: ";
        data_file << std::setprecision(12);
        data_file << SSV::eval(slip_np1[s]) << " ";
      }

      data_file << "\n" << "F: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << SSV::eval(F_np1(i,j)) << " ";
        }
      }

      data_file << "\n" << "Fp: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << SSV::eval(Fp_np1(i,j)) << " ";
        }
      }

      data_file << "\n" << "Sigma: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << SSV::eval(sigma_np1(i,j)) << " ";
        }
      }

      data_file << "\n" << "Lp: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << SSV::eval(Lp_np1(i,j)) << " ";
        }
      }
      data_file << "\n";
      data_file.close();
    }
  } // end data file output
} // computeState


///
/// Return calculated quantities to Albany
///
template<typename EvalT, typename Traits>
void
CrystalPlasticityKernel<EvalT, Traits>::finalize(
    CP::StateMechanical<ScalarT, CP::MAX_DIM> const & state_mechanical,
    CP::StateInternal<ScalarT, CP::MAX_SLIP> const & state_internal,
    utility::StaticPointer<CP::Integrator<EvalT, CP::MAX_DIM, CP::MAX_SLIP>> const & integrator,
    int const cell,
    int const pt) const
{
  ///
  /// Mechanical state
  ///
  minitensor::Tensor<RealType, CP::MAX_DIM> const
  F_n = state_mechanical.F_n_;

  minitensor::Tensor<ScalarT, CP::MAX_DIM> const
  F_np1 = state_mechanical.F_np1_;

  minitensor::Tensor<ScalarT, CP::MAX_DIM> const
  Fp_np1 = state_mechanical.Fp_np1_;

  minitensor::Tensor<ScalarT, CP::MAX_DIM> const
  Lp_np1 = state_mechanical.Lp_np1_;

  minitensor::Tensor<ScalarT, CP::MAX_DIM> const
  sigma_np1 = state_mechanical.sigma_np1_;

  ///
  /// Internal state
  ///
  minitensor::Vector<ScalarT, CP::MAX_SLIP> const
  state_hardening_np1 = state_internal.hardening_np1_;

  minitensor::Vector<ScalarT, CP::MAX_SLIP> const
  slip_np1 = state_internal.slip_np1_;

  minitensor::Vector<ScalarT, CP::MAX_SLIP> const
  shear_np1 = state_internal.shear_np1_;

  minitensor::Vector<ScalarT, CP::MAX_SLIP> const
  rates_slip = state_internal.rates_slip_;

  ///
  /// Mechanical heat source
  ///
  if (have_temperature_ == true)
  {
    ScalarT const
    plastic_dissipation = minitensor::dot(rates_slip, shear_np1);

    source_(cell, pt) = 0.9 / (density_ * heat_capacity_) * plastic_dissipation;
  }

  ///
  /// Compute the equivalent plastic strain from the plastic velocity gradient:
  ///  eqps_dot = sqrt[2/3* sym(Lp) : sym(Lp)]
  ///
  minitensor::Tensor<ScalarT, CP::MAX_DIM> const
  Dp = minitensor::sym(Lp_np1);

  eqps_(cell, pt) += dt_ * std::sqrt(2.0 / 3.0 * SSV::eval(minitensor::dotdot(Dp,Dp)));


// The xtal rotation from the polar decomp of Fe.
  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  Fe(num_dims_);

  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  Re_np1(num_dims_);

  // Multiplicatively decompose F to get Fe
  Fe = F_np1 * minitensor::inverse(Fp_np1);

  // Compute polar rotation to get Re
  Re_np1 = minitensor::polar_rotation(Fe);

  ///
  /// Copy data from local data structures back into Albany fields
  ///

  // residual norm
  cp_residual_(cell, pt) = integrator->getNormResidual();
  cp_residual_iter_(cell,pt) = integrator->getNumIters();

  minitensor::Tensor<RealType, CP::MAX_DIM> const
  inv_F = minitensor::inverse(F_n);

  minitensor::Tensor<RealType, CP::MAX_DIM> const
  eye = minitensor::identity<RealType, CP::MAX_DIM>(num_dims_);


  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  L(num_dims_, minitensor::Filler::ZEROS);

  if (dt_ > 0.0) {
    L = 1.0 / dt_ * (F_np1 * inv_F - eye);
  }

  // num_dims_ x num_dims_ dimensional array variables
  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      xtal_rotation_(cell, pt, i, j) = Re_np1(i, j);
      plastic_deformation_(cell, pt, i, j) = Fp_np1(i, j);
      stress_(cell, pt, i, j) = sigma_np1(i, j);
      velocity_gradient_(cell, pt, i, j) = L(i, j);
      velocity_gradient_plastic_(cell, pt, i, j) = Lp_np1(i, j);
    }
  }

  // num_slip_ dimensional array variables
  for (int s(0); s < num_slip_; ++s) {
    (*(slips_[s]))(cell, pt) = slip_np1[s];
    (*(hards_[s]))(cell, pt) = state_hardening_np1[s];
    (*(shears_[s]))(cell, pt) = shear_np1[s];
    (*(slip_rates_[s]))(cell, pt) = state_internal.rates_slip_[s];
  }

  return;
} // void finalize

} // namespace LCM
