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
    element_block_orientation_.set_dimension(num_dims_);
    for (int i = 0; i < num_dims_; ++i) {
      std::vector<RealType> const
      b_temp = e_list.get<Teuchos::Array<RealType>>(
        Albany::strint("Basis Vector", i + 1)).toVector();

      minitensor::Vector<RealType, CP::MAX_DIM>
      basis(num_dims_);

      for (int dim = 0; dim < num_dims_; ++dim){
        basis[dim] = b_temp[dim];
      }

      basis = minitensor::unit(basis);

      // TODO check zero, rh system
      // Filling columns of transformation with basis vectors
      // We are forming R^{T} which is equivalent to the direction cosine matrix
      for (int j = 0; j < num_dims_; ++j) {
        element_block_orientation_(j, i) = basis[j];
      }
    }
  }

	integration_scheme_ = preader.getIntegrationScheme();
  residual_type_ = preader.getResidualType();
	step_type_ = preader.getStepType();
	minimizer_ = preader.getMinimizer();
  predictor_slip_ = preader.getPredictorSlip();

  verbosity_ = p->get<int>("Verbosity", 0);

  write_data_file_ = p->get<bool>("Write Data File", false);

  if (verbosity_ > 2) {
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
  c12_temperature_coeff_ = e_list.get<RealType>("M13", NAN);
  c12_temperature_coeff_ = e_list.get<RealType>("M33", NAN);
  c44_temperature_coeff_ = e_list.get<RealType>("M44", NAN);
  reference_temperature_ = e_list.get<RealType>("Reference Temperature", NAN);

  C_unrotated_.set_dimension(num_dims_);
  if (c11_ == c33_) {
    c66_ = c44_;
    c66_temperature_coeff_ = c44_temperature_coeff_;
  } else {
    c66_ = 0.5 * (c11_ - c12_);
    c66_temperature_coeff_ = 
      0.5 * (c11_temperature_coeff_ - c12_temperature_coeff_);
  }

  CP::computeElasticityTensor(c11_, c12_, c13_, c33_, c44_, c66_, C_unrotated_);

  if (verbosity_ > 2) {
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
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

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
    s_temp_normalized(num_dims_);

    for (int i = 0; i < num_dims_; ++i) {
      s_temp_normalized[i] = s_temp[i];
    }
    s_temp_normalized = minitensor::unit(s_temp_normalized);
    slip_systems_.at(num_ss).s_.set_dimension(num_dims_);
    slip_systems_.at(num_ss).s_ = s_temp_normalized;

    //
    // Read and normalize slip normals. Miller indices need to be normalized.
    //
    std::vector<RealType> 
    n_temp = ss_list.get<Teuchos::Array<RealType>>("Slip Normal").toVector();

    minitensor::Vector<RealType, CP::MAX_DIM>
    n_temp_normalized(num_dims_);

    for (int i = 0; i < num_dims_; ++i) {
      n_temp_normalized[i] = n_temp[i];
    }

    n_temp_normalized = minitensor::unit(n_temp_normalized);
    slip_systems_.at(num_ss).n_.set_dimension(num_dims_);
    slip_systems_.at(num_ss).n_ = n_temp_normalized;

    slip_systems_.at(num_ss).projector_.set_dimension(num_dims_);
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

  for (int sf_index(0); sf_index < num_family_; ++sf_index) {
    auto &
    slip_family = slip_families_[sf_index];

    // Create latent matrix for hardening law
    slip_family.phardening_parameters_->createLatentMatrix(
      slip_family, slip_systems_); 

    if (verbosity_ > 2) {
      std::cout << slip_family.latent_matrix_ << std::endl;
    }

    slip_family.slip_system_indices_.set_dimension(slip_family.num_slip_sys_);

    if (verbosity_ > 2) {
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
  setEvaluatedField(source_string_, dl->qp_scalar);
  setEvaluatedField(residual_string_, dl->qp_scalar);
  setEvaluatedField(residual_iter_string_, dl->qp_scalar);

  if (have_temperature_) {
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
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

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
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

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
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

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
  if(verbosity_ == 99) {
    index_element_ = workset.wsIndex;
  }
  else{
    index_element_ = -1;
  }
  
  if(verbosity_ > 2) {
    std::cout << ">>> in cp initialize compute state\n";
  }

  if (read_orientations_from_mesh_) {
    rotation_matrix_transpose_ = workset.wsLatticeOrientation;

    TEUCHOS_TEST_FOR_EXCEPTION(
      rotation_matrix_transpose_.is_null(),
      std::logic_error,
      "\n**** Error in CrystalPlasticityModel: \
         rotation matrix not found on genesis mesh.\n");
  }

  //
  // extract dependent MDFields
  //
  def_grad_ = *dep_fields[F_string_];
  if (write_data_file_) {
    time_ = *dep_fields[time_string_];
  }
  delta_time_ = *dep_fields[dt_string_];

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
  
  if (have_temperature_) {
    source_ = *eval_fields[source_string_];
  }

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
  
  dt_ = Sacado::ScalarValue<ScalarT>::eval(delta_time_(0));

  // Resest status and status message for model failure test
  nox_status_test_->status_message_ = "";
  nox_status_test_->status_ = NOX::StatusTest::Unevaluated;
}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
CrystalPlasticityKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  // TODO: In the future for CUDA this should be moved out of the kernel because
  // it uses dynamic allocation for the buffer. It should also be modified to use 
  // cudaMalloc.
  utility::StaticAllocator 
  allocator(1024 * 1024);

  if (nox_status_test_->status_ == NOX::StatusTest::Failed)
  {
    return;
  }

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

  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  slip_resistance(num_slip_);

  minitensor::Vector<ScalarT, CP::MAX_SLIP>
  slip_computed(num_slip_);

  ///
  /// Elasticity tensor
  ///
  minitensor::Tensor4<ScalarT, CP::MAX_DIM>
  C_unrotated = C_unrotated_;

  minitensor::Tensor4<ScalarT, CP::MAX_DIM>
  C(num_dims_);

  RealType
  norm_slip_residual;

  RealType
  residual_iter;

  RealType
  equivalent_plastic_strain;

  bool
  update_state_successful{true};

  minitensor::Tensor<RealType, CP::MAX_DIM>
  orientation_matrix(num_dims_);

  std::vector<CP::SlipSystem<CP::MAX_DIM>>
  element_slip_systems = slip_systems_;

  if (have_temperature_) {

    RealType const
    tlocal = Sacado::ScalarValue<ScalarT>::eval(temperature_(cell,pt));

    RealType const
    c11 = c11_ + c11_temperature_coeff_ * (tlocal - reference_temperature_);

    RealType const
    c12 = c12_ + c12_temperature_coeff_ * (tlocal - reference_temperature_);

    RealType const
    c13 = c13_ + c13_temperature_coeff_ * (tlocal - reference_temperature_);

    RealType const
    c33 = c33_ + c44_temperature_coeff_ * (tlocal - reference_temperature_);

    RealType const
    c44 = c44_ + c44_temperature_coeff_ * (tlocal - reference_temperature_);

    RealType const
    c66 = c66_ + c44_temperature_coeff_ * (tlocal - reference_temperature_);

    CP::computeElasticityTensor(c11, c12, c13, c33, c44, c66, C_unrotated);

    if (verbosity_ > 2) {
      std::cout << "tlocal: " << tlocal << std::endl;
      std::cout << "c11, c12, c44: " << c11 << c12 << c44 << std::endl;
    }
  }

  if (read_orientations_from_mesh_) {
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        orientation_matrix(i,j) = rotation_matrix_transpose_[cell][i * 3 + j];
      }
    }
  }
  else {
    orientation_matrix = element_block_orientation_;
  }

  // Set the rotated elasticity tensor, slip normals, slip directions, 
  // and projection operator
  C = minitensor::kronecker(orientation_matrix, C_unrotated);
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    element_slip_systems.at(num_ss).s_ = 
      orientation_matrix * slip_systems_.at(num_ss).s_;
    element_slip_systems.at(num_ss).n_ = 
      orientation_matrix * slip_systems_.at(num_ss).n_;
    element_slip_systems.at(num_ss).projector_ =
      minitensor::dyad(element_slip_systems.at(num_ss).s_,
                      element_slip_systems.at(num_ss).n_);
  }

  equivalent_plastic_strain = 
    Sacado::ScalarValue<ScalarT>::eval(eqps_(cell, pt));

  // Copy data from Albany fields into local data structures
  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      F_np1(i, j) = def_grad_(cell, pt, i, j);
      Fp_n(i, j) = previous_plastic_deformation_(cell, pt, i, j);
      F_n(i, j) = previous_defgrad_(cell, pt, i, j);
    }
  }

  // Bring in internal state from previous step
  for (int s(0); s < num_slip_; ++s)
  {
    slip_n[s] = (*(previous_slips_[s]))(cell, pt);
    slip_np1[s] = slip_n[s];
    slip_dot_n[s] = (*(previous_slip_rates_[s]))(cell, pt);
    state_hardening_n[s] = (*(previous_hards_[s]))(cell, pt);
  }

  //
  // Set up slip predictor to assign isochoric part of F_increment to Fp_increment
  //
  if (dt_ > 0.0)
  {
    switch (predictor_slip_)
    {
      case CP::PredictorSlip::RATE:
      {
        for (int s(0); s < num_slip_; ++s)
        {
          slip_np1[s] += dt_ * slip_dot_n[s];
        }
      } break;

      case CP::PredictorSlip::SOLVE:
      {
        auto const
        size_problem = std::max(num_slip_, num_dims_ * num_dims_);

        minitensor::Tensor<RealType, CP::MAX_SLIP>
        dyad_matrix(size_problem);

        dyad_matrix.fill(minitensor::ZEROS);

        for (int s = 0; s < num_slip_; ++s)
        {
          for (int d(0); d < num_dims_ * num_dims_; ++d)
          {
            dyad_matrix(d, s) = element_slip_systems.at(s).projector_[d];
          }
        }

        minitensor::Tensor<RealType, CP::MAX_SLIP>
        A(size_problem);
        minitensor::Tensor<RealType, CP::MAX_SLIP>
        B(size_problem);
        minitensor::Tensor<RealType, CP::MAX_SLIP>
        C(size_problem);

        boost::tie(A, B, C) = minitensor::svd(dyad_matrix);

        for (int s(0); s < num_slip_; ++s)
        {
          B(s, s) = B(s, s) > 1.0e-12 ? 1.0 / B(s,s) : 0.0;
        }

        minitensor::Tensor<RealType, CP::MAX_SLIP>
        Pinv(num_slip_);

        // Pinv = C * B * minitensor::transpose(A);
        Pinv = C * B * B * minitensor::transpose(C);

        minitensor::Tensor<RealType, CP::MAX_DIM> const
        inv_F = minitensor::inverse(F_n);

        minitensor::Tensor<RealType, CP::MAX_DIM> const
        eye = minitensor::identity<RealType, CP::MAX_DIM>(num_dims_);

        minitensor::Tensor<ScalarT, CP::MAX_DIM>
        L(num_dims_);

        L = 1.0 / dt_ * (F_np1 * inv_F - eye);

        minitensor::Vector<RealType, CP::MAX_SLIP>
        L_vec(size_problem);

        L_vec.fill(minitensor::ZEROS);

        RealType
        portion_L = 0.5;

        for (int i = 0; i < num_dims_; ++i)
        {
          for (int j = 0; j < num_dims_; ++j)
          {
            L_vec(i * num_dims_ + j) = portion_L * Sacado::ScalarValue<ScalarT>::eval(L(i, j));
          }
        }

        minitensor::Vector<RealType, CP::MAX_SLIP>
        dm_lv = minitensor::transpose(dyad_matrix) * L_vec;

        minitensor::Vector<RealType, CP::MAX_SLIP>
        rates_slip_trial = Pinv * dm_lv;

        for (int s(0); s < num_slip_; ++s)
        {
          slip_np1[s] = slip_n[s] + dt_ * rates_slip_trial[s];
        }

        if (verbosity_ > 4)
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

        minitensor::Tensor<ScalarT, CP::MAX_DIM>
        Lp_trial(num_dims_);
        Lp_trial.fill(minitensor::ZEROS);
        // for (int s(0); s < num_slip_; ++s)
        // {
        //   Lp_trial += rates_slip_trial[s] * element_slip_systems.at(s).projector_;
        // }

        minitensor::Vector<RealType, CP::MAX_SLIP>
        Lp_vec = dyad_matrix * rates_slip_trial;

        for (int i = 0; i < num_dims_; ++i)
        {
          for (int j = 0; j < num_dims_; ++j)
          {
            Lp_trial(i, j) = Lp_vec(i * num_dims_ + j);
          }
        }

        if (verbosity_ > 4)
        {
          std::cout << "L" << std::endl;
          std::cout << std::setprecision(4) << L << std::endl;

          std::cout << "Lp_trial" << std::endl;
          std::cout << std::setprecision(4) << Lp_trial << std::endl;

          std::cout << "exp(Lp * dt_)" << std::endl;
          std::cout << std::setprecision(4) << minitensor::exp(dt_ * Lp_trial)<< std::endl;
        }

      } break;

      default:
      {
        
      } break;
    }
  }

  if(verbosity_ > 2) {
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
  state_mechanical(num_dims_, Fp_n);

  CP::StateInternal<ScalarT, CP::MAX_SLIP>
  state_internal(index_element_, pt, num_slip_, state_hardening_n, slip_n);

  if (dt_ > 0.0) {
    state_internal.rate_slip_ = (slip_np1 - slip_n) / dt_;
  }
  else {
    state_internal.rate_slip_.fill(minitensor::ZEROS);
  }

  state_internal.slip_np1_ = slip_np1;

  bool
  failed{false};
  
  if (dt_ == 0.0)
  {
    if (verbosity_ == 99) {
      std::ofstream outfile;
      std::stringstream ss;
      ss << "slips_" << index_element_
	 << "_" << pt <<  ".out";
      std::string file = ss.str();
      outfile.open(file);
      outfile.close();
    }
        
  }  
  auto
  integratorFactory = CP::IntegratorFactory<EvalT, CP::MAX_DIM, CP::MAX_SLIP>(
    allocator,
    minimizer_,
    step_type_,
    nox_status_test_,
    element_slip_systems,
    slip_families_,
    state_mechanical,
    state_internal,
    C,
    F_n,
    F_np1,
    dt_,
    failed);

  auto
  integrator = integratorFactory(integration_scheme_, residual_type_);

  update_state_successful = integrator->update(norm_slip_residual);
  
  residual_iter = integrator->getNumIters();

  Fp_np1 = state_mechanical.Fp_np1_;
  Lp_np1 = state_mechanical.Lp_np1_;
  sigma_np1 = state_mechanical.sigma_np1_;
  S_np1 = state_mechanical.S_np1_;

  state_hardening_np1 = state_internal.hardening_np1_;
  slip_resistance = state_internal.resistance_;
  slip_np1 = state_internal.slip_np1_;
  shear_np1 = state_internal.shear_np1_;

  // Exit early if update state is not successful
  if(!update_state_successful){
    return;
  }

  // Compute the equivalent plastic strain from the plastic velocity gradient:
  //  eqps_dot = sqrt[2/3* sym(Lp) : sym(Lp)]
  minitensor::Tensor<ScalarT, CP::MAX_DIM> const
  Dp = minitensor::sym(Lp_np1);

  RealType const
  delta_eqps = dt_ * std::sqrt(2.0 / 3.0 *
    Sacado::ScalarValue<ScalarT>::eval(minitensor::dotdot(Dp,Dp)));

  equivalent_plastic_strain += delta_eqps;

  eqps_(cell, pt) = equivalent_plastic_strain;

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

  // mechanical heat source
  if (have_temperature_) {
    source_(cell, pt) = 0.0;
    if (dt_ > 0.0) {

      RealType
      plastic_dissipation(0.0);

      for (int slip_system(0); slip_system < num_slip_; ++slip_system) {
        plastic_dissipation += Sacado::ScalarValue<ScalarT>::eval(
          state_internal.rate_slip_[slip_system] * shear_np1[slip_system]);
      }
      source_(cell, pt) = 0.9 / (density_ * heat_capacity_) * plastic_dissipation;
    }
  }

  // residual norm
  cp_residual_(cell, pt) = norm_slip_residual;
  cp_residual_iter_(cell,pt) = residual_iter;

  minitensor::Tensor<RealType, CP::MAX_DIM> const
  inv_F = minitensor::inverse(F_n);

  minitensor::Tensor<RealType, CP::MAX_DIM> const
  eye = minitensor::identity<RealType, CP::MAX_DIM>(num_dims_);


  minitensor::Tensor<ScalarT, CP::MAX_DIM>
  L(num_dims_);

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
    (*(slip_rates_[s]))(cell, pt) = state_internal.rate_slip_[s];
  }

  if(write_data_file_) {
    if (cell == 0 && pt == 0) {

      std::ofstream
      data_file("output.dat", std::fstream::app);

      minitensor::Tensor<RealType, CP::MAX_DIM>
      P(num_dims_);

      data_file << "\n" << "time: ";
      data_file << std::setprecision(12);
      data_file << Sacado::ScalarValue<ScalarT>::eval(time_(0));
      data_file << "     dt: ";
      data_file << std::setprecision(12) << dt_ << " \n";

      for (int s(0); s < num_slip_; ++s) {
        data_file << "\n" << "P" << s << ": ";
        P = element_slip_systems.at(s).projector_;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            data_file << std::setprecision(12);
            data_file << Sacado::ScalarValue<ScalarT>::eval(P(i,j)) << " ";
          }
        }
      }
      for (int s(0); s < num_slip_; ++s) {
        data_file << "\n" << "slips: ";
        data_file << std::setprecision(12);
        data_file << Sacado::ScalarValue<ScalarT>::eval(slip_np1[s]) << " ";
      }
      data_file << "\n" << "F: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << Sacado::ScalarValue<ScalarT>::eval(F_np1(i,j)) << " ";
        }
      }
      data_file << "\n" << "Fp: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << Sacado::ScalarValue<ScalarT>::eval(Fp_np1(i,j)) << " ";
        }
      }
      data_file << "\n" << "Sigma: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << Sacado::ScalarValue<ScalarT>::eval(sigma_np1(i,j)) << " ";
        }
      }
      data_file << "\n" << "Lp: ";
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          data_file << std::setprecision(12);
          data_file << Sacado::ScalarValue<ScalarT>::eval(Lp_np1(i,j)) << " ";
        }
      }
      data_file << "\n";
      data_file.close();
    }
  } // end data file output
} // computeState

} // namespace LCM
