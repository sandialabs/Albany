//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

#include <MiniLinearSolver.h>

#include <typeinfo>
#include <iostream>
#include <Sacado_Traits.hpp>

#include "core/CrystalPlasticity/ParameterReader.hpp"
#include "core/CrystalPlasticity/Integrator.hpp"

namespace LCM
{

template<typename EvalT, typename Traits>
CrystalPlasticityModel<EvalT, Traits>::
CrystalPlasticityModel(
    Teuchos::ParameterList* p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    num_family_(p->get<int>("Number of Slip Families", 1)),
    num_slip_(p->get<int>("Number of Slip Systems", 0)),
    allocator_(1024*1024)  // 1 mB
{
	CP::ParameterReader<EvalT, Traits> preader(p, allocator_);

  slip_systems_.resize(num_slip_);

  // Store an RCP to the NOX status test, if available
  if (p->isParameter("NOX Status Test")) {
    nox_status_test_ = 
        p->get<Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>>(
            "NOX Status Test");
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
      std::vector<RealType> 
      b_temp = e_list.get<Teuchos::Array<RealType>>(
        Albany::strint("Basis Vector", i + 1)).toVector();

      RealType 
      norm{0.};

      for (int j = 0; j < num_dims_; ++j) {
        norm += b_temp[j] * b_temp[j];
      }

      RealType const 
      inverse_norm = 1. / std::sqrt(norm);

      // TODO check zero, rh system
      // Filling columns of transformation with basis vectors
      // We are forming R^{T} which is equivalent to the direction cosine matrix
      for (int j = 0; j < num_dims_; ++j) {
        element_block_orientation_(j, i) = b_temp[j] * inverse_norm;
      }
    }
  }

	integration_scheme_ = preader.getIntegrationScheme();
  residual_type_ = preader.getResidualType();
	step_type_ = preader.getStepType();
	minimizer_ = preader.getMinimizer();

  apply_slip_predictor_ = p->get<bool>("Apply Slip Predictor", true);

  verbosity_ = p->get<int>("Verbosity", 0);

  write_data_file_ = p->get<bool>("Write Data File", false);

  if (verbosity_ > 2) {
    std::cout << ">>> in cp constructor\n";
    std::cout << ">>> parameter list:\n" << *p << std::endl;
  }

  //
  // Obtain crystal elasticity constants and populate elasticity tensor
  // assuming cubic symmetry
  //
  c11_ = e_list.get<RealType>("C11");
  c12_ = e_list.get<RealType>("C12");
  c44_ = e_list.get<RealType>("C44");
  c11_temperature_coeff_ = e_list.get<RealType>("M11",0.0);
  c12_temperature_coeff_ = e_list.get<RealType>("M12",0.0);
  c44_temperature_coeff_ = e_list.get<RealType>("M44",0.0);
  reference_temperature_ = e_list.get<RealType>("Reference Temperature",0.0);

  C_unrotated_.set_dimension(num_dims_);
  CP::computeCubicElasticityTensor(c11_, c12_, c44_, C_unrotated_);
  C_.set_dimension(num_dims_);

  if (verbosity_ > 2) {
    // print elasticity tensor
    std::cout << ">>> Unrotated C :" << std::endl << C_unrotated_ << std::endl;
  }


  //
  // Get slip families.
  //
  slip_families_.reserve(num_family_);
  for (int num_fam(0); num_fam < num_family_; ++num_fam) {
    slip_families_.emplace_back(preader.getSlipFamily(num_fam));
  }

  //
  // Get slip system information
  //
  Intrepid2::Vector<RealType, CP::MAX_SLIP>
  state_hardening_initial(num_slip_);

  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

    Teuchos::ParameterList
    ss_list = p->sublist(Albany::strint("Slip System", num_ss + 1));

    CP::SlipSystem<CP::MAX_DIM> &
    slip_system = slip_systems_.at(num_ss);

    slip_system.slip_family_index_ = ss_list.get<int>("Slip Family", 0);

    CP::SlipFamily<CP::MAX_DIM, CP::MAX_SLIP> &
    slip_family = slip_families_[slip_system.slip_family_index_];

    Intrepid2::Index 
    slip_system_index = slip_family.num_slip_sys_;

    slip_family.slip_system_indices_[slip_system_index] = num_ss;

    slip_family.num_slip_sys_ = slip_system_index + 1;

    //
    // Read and normalize slip directions. Miller indices need to be normalized.
    //
    std::vector<RealType>
    s_temp = ss_list.get<Teuchos::Array<RealType>>("Slip Direction").toVector();

    Intrepid2::Vector<RealType, CP::MAX_DIM>
    s_temp_normalized(num_dims_);

    for (int i = 0; i < num_dims_; ++i) {
      s_temp_normalized[i] = s_temp[i];
    }
    s_temp_normalized = Intrepid2::unit(s_temp_normalized);
    slip_systems_.at(num_ss).s_.set_dimension(num_dims_);
    s_unrotated_.push_back( s_temp_normalized );

    //
    // Read and normalize slip normals. Miller indices need to be normalized.
    //
    std::vector<RealType> 
    n_temp = ss_list.get<Teuchos::Array<RealType>>("Slip Normal").toVector();

    Intrepid2::Vector<RealType, CP::MAX_DIM>
    n_temp_normalized(num_dims_);

    for (int i = 0; i < num_dims_; ++i) {
      n_temp_normalized[i] = n_temp[i];
    }
    n_temp_normalized = Intrepid2::unit(n_temp_normalized);
    slip_systems_.at(num_ss).n_.set_dimension(num_dims_);
    n_unrotated_.push_back( n_temp_normalized );

    slip_systems_.at(num_ss).projector_.set_dimension(num_dims_);

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

    slip_family.slip_system_indices_.set_dimension(slip_family.num_slip_sys_);

    if (verbosity_ > 2) {
      std::cout << "slip system indices" << slip_family.slip_system_indices_ << std::endl;
    }
  }

  //
  // retrive appropriate field name strings (ref to problems/FieldNameMap)
  //
  std::string const
  eqps_string = (*field_name_map_)["eqps"];

  std::string const
  Re_string = (*field_name_map_)["Re"];

  std::string const
  cauchy_string = (*field_name_map_)["Cauchy_Stress"];

  std::string const
  Fp_string = (*field_name_map_)["Fp"];

  std::string const
  L_string = (*field_name_map_)["Velocity_Gradient"];

  std::string const
  F_string = (*field_name_map_)["F"];

  std::string const
  J_string = (*field_name_map_)["J"];

  std::string const
  source_string = (*field_name_map_)["Mechanical_Source"];

  std::string const
  residual_string = (*field_name_map_)["CP_Residual"];

  std::string const
  residual_iter_string = (*field_name_map_)["CP_Residual_Iter"];

  //
  // define the dependent fields required for calculation
  //
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  //
  // define the evaluated fields
  //
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(Re_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(L_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(residual_string, dl->qp_scalar));
  this->eval_field_map_.insert(
    std::make_pair(residual_iter_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Time", dl->workset_scalar));

  if (have_temperature_) {
    this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  }

  //
  // define the state variables
  //

  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output eqps", false));

  // Re
  this->num_state_variables_++;
  this->state_var_names_.push_back(Re_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Re", false));

  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));

  // Fp
  this->num_state_variables_++;
  this->state_var_names_.push_back(Fp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Fp", false));

  // L
  this->num_state_variables_++;
  this->state_var_names_.push_back(L_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output L", false));

  // mechanical source
  if (have_temperature_) {
    this->num_state_variables_++;
    this->state_var_names_.push_back(source_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(
        p->get<bool>("Output Mechanical Source", false));
  }

  // gammas for each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

    std::string const
    g = Albany::strint("gamma", num_ss + 1, '_');

    std::string const
    gamma_string = (*field_name_map_)[g];

    std::string const
    output_gamma_string = "Output " + gamma_string;

    this->eval_field_map_.insert(std::make_pair(gamma_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(gamma_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_gamma_string, false));
  }

  // gammadots for each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

    std::string const
    g_dot = Albany::strint("gamma_dot", num_ss + 1, '_');

    std::string const
    gamma_dot_string = (*field_name_map_)[g_dot];

    std::string const
    output_gamma_dot_string = "Output " + gamma_dot_string;

    this->eval_field_map_.insert(
        std::make_pair(gamma_dot_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(gamma_dot_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_gamma_dot_string, false));
  }

  // tau_hard - state variable for hardening on each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

    std::string const
    t_h = Albany::strint("tau_hard", num_ss + 1, '_');

    std::string const
    tau_hard_string = (*field_name_map_)[t_h];

    std::string const
    output_tau_hard_string = "Output " + tau_hard_string;

    this->eval_field_map_.insert(
        std::make_pair(tau_hard_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(tau_hard_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(
      slip_systems_.at(num_ss).state_hardening_initial_);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_tau_hard_string, false));
  }

  // taus - output resolved shear stress for debugging - not stated
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) 
  {
    std::string const
    t = Albany::strint("tau", num_ss + 1, '_');

    std::string const
    tau_string = (*field_name_map_)[t];

    std::string const
    output_tau_string = "Output " + tau_string;

    this->eval_field_map_.insert(std::make_pair(tau_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(tau_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_tau_string, false));
  }

  // residual
  this->num_state_variables_++;
  this->state_var_names_.push_back(residual_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output CP_Residual", false));

  // residual iterations
  this->num_state_variables_++;
  this->state_var_names_.push_back(residual_iter_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output CP_Residual_Iter", false));    

}








//
// Compute the constitutive response of the material
//
template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
{
  allocator_.clear();


  if(verbosity_ > 2) {
    std::cout << ">>> in cp compute state\n";
  }

  if (read_orientations_from_mesh_) {
    Teuchos::ArrayRCP<double*> const &
    rotation_matrix_transpose = workset.wsLatticeOrientation;

    TEUCHOS_TEST_FOR_EXCEPTION(
      rotation_matrix_transpose.is_null(),
      std::logic_error,
      "\n**** Error in CrystalPlasticityModel: \
         rotation matrix not found on genesis mesh.\n");
  }

  //
  // retrive appropriate field name strings
  //
  std::string const
  eqps_string = (*field_name_map_)["eqps"];

  std::string const
  Re_string = (*field_name_map_)["Re"];

  std::string const
  cauchy_string = (*field_name_map_)["Cauchy_Stress"];

  std::string const
  Fp_string = (*field_name_map_)["Fp"];

  std::string const
  L_string = (*field_name_map_)["Velocity_Gradient"];

  std::string const
  residual_string = (*field_name_map_)["CP_Residual"];

  std::string const
  residual_iter_string = (*field_name_map_)["CP_Residual_Iter"];

  std::string const
  source_string = (*field_name_map_)["Mechanical_Source"];

  std::string const
  F_string = (*field_name_map_)["F"];

  std::string const
  J_string = (*field_name_map_)["J"];

  //
  // extract dependent MDFields
  //
  PHX::MDField<ScalarT>
  def_grad = *dep_fields[F_string];

  PHX::MDField<ScalarT>
  delta_time = *dep_fields["Delta Time"];

  //
  // extract evaluated MDFields
  //
  PHX::MDField<ScalarT>
  eqps = *eval_fields[eqps_string];

  PHX::MDField<ScalarT>
  xtal_rotation = *eval_fields[Re_string];

  PHX::MDField<ScalarT>
  stress = *eval_fields[cauchy_string];

  PHX::MDField<ScalarT>
  plastic_deformation = *eval_fields[Fp_string];

  PHX::MDField<ScalarT>
  velocity_gradient = *eval_fields[L_string];

  PHX::MDField<ScalarT>
  source;

  if (have_temperature_) {
    source = *eval_fields[source_string];
  }

  PHX::MDField<ScalarT>
  cp_residual = *eval_fields[residual_string];

  PHX::MDField<ScalarT>
  cp_residual_iter = *eval_fields[residual_iter_string];

  PHX::MDField<ScalarT>
  time = *eval_fields["Time"];

  // extract slip on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>>
  slips;

  std::vector<Albany::MDArray *>
  previous_slips;

  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    g = Albany::strint("gamma", num_ss + 1, '_');

    std::string const
    gamma_string = (*field_name_map_)[g];

    slips.push_back(eval_fields[gamma_string]);
    previous_slips.push_back(
    &((*workset.stateArrayPtr)[gamma_string + "_old"]));
  }

  // extract slip rate on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>>
  rate_slip_np1;

  std::vector<Albany::MDArray *>
  rate_slip_n;

  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {

    std::string const
    g_dot = Albany::strint("gamma_dot", num_ss + 1, '_');

    std::string const
    gamma_dot_string = (*field_name_map_)[g_dot];

    rate_slip_np1.push_back(eval_fields[gamma_dot_string]);
    rate_slip_n.push_back(
    &((*workset.stateArrayPtr)[gamma_dot_string + "_old"]));
  }

  // extract hardening on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>>
  hards;

  std::vector<Albany::MDArray *>
  state_hardening;

  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    t_h = Albany::strint("tau_hard", num_ss + 1, '_');

    std::string const
    tau_hard_string = (*field_name_map_)[t_h];

    hards.push_back(eval_fields[tau_hard_string]);

    state_hardening.push_back(
    &((*workset.stateArrayPtr)[tau_hard_string + "_old"]));
  }

  // store shear on each slip system for output
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>>
  shears;

  for (int num_ss = 0; num_ss < num_slip_; ++num_ss)
  {
    std::string const
    t = Albany::strint("tau", num_ss + 1, '_');

    std::string const
    tau_string = (*field_name_map_)[t];

    shears.push_back(eval_fields[tau_string]);
  }

  // get state variables

  Albany::MDArray
  previous_plastic_deformation = (*workset.stateArrayPtr)[Fp_string + "_old"];

  RealType
  dt = Sacado::ScalarValue<ScalarT>::eval(delta_time(0));

  // -- Local variables for implicit integration routine --

  //
  // Known quantities
  //
  Intrepid2::Tensor<RealType, CP::MAX_DIM> 
  Fp_n(num_dims_);

  Intrepid2::Vector<RealType, CP::MAX_SLIP>
  slip_n(num_slip_);

  Intrepid2::Vector<RealType, CP::MAX_SLIP>
  slip_dot_n(num_slip_);

  Intrepid2::Vector<RealType, CP::MAX_SLIP>
  state_hardening_n(num_slip_);

  Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
  F_np1(num_dims_);

  //
  // Unknown quantities
  //
  Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
  Lp_np1(num_dims_);

  Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
  Fp_np1(num_dims_);

  Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
  sigma_np1(num_dims_);

  Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
  S_np1(num_dims_);

  Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
  rate_slip(num_slip_);

  Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
  slip_np1(num_slip_);

  Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
  shear_np1(num_slip_);

  Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
  state_hardening_np1(num_slip_);

  Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
  slip_resistance(num_slip_);

  Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
  slip_computed(num_slip_);

  RealType
  norm_slip_residual;

  RealType
  residual_iter;

  RealType
  equivalent_plastic_strain;

  bool
  update_state_successful{true};

  Intrepid2::Tensor<RealType, CP::MAX_DIM>
  orientation_matrix(num_dims_);

  // unknowns array
  Intrepid2::Vector<ScalarT, CP::NLS_DIM>
  x;

  // Choose explicit or implicit integration of the constitutive update
  switch (integration_scheme_) {

    default:
    break;

		case CP::IntegrationScheme::EXPLICIT:
    {
      // unknowns, which are slip_np1
      x.set_dimension(num_slip_);          
    }
    break;

		case CP::IntegrationScheme::IMPLICIT:
    {
      //
      // Chose residual type
      //
      switch (residual_type_)
      {
				case CP::ResidualType::SLIP:
        {
          // unknowns, which are slip_np1
          x.set_dimension(num_slip_);
        }
        break;

				case CP::ResidualType::SLIP_HARDNESS:
        {
          // unknowns, which are slip_np1 followed by slip_resistance
          x.set_dimension(2 * num_slip_);
        }
        break;

        default:
          std::cerr << "You must supply a residual type." << std::endl;
          exit(1);
        break;
      }
    }
  }

  x.fill(Intrepid2::ZEROS);
















  //
  // Material point loop
  //
  for (int cell(0); cell < workset.numCells; ++cell) {

    for (int pt(0); pt < num_pts_; ++pt) {

      allocator_.clear();

      if (have_temperature_) {

        RealType const
        tlocal = Sacado::ScalarValue<ScalarT>::eval(temperature_(cell,pt));

        RealType const
        c11 = c11_ + c11_temperature_coeff_ * (tlocal - reference_temperature_);

        RealType const
        c12 = c12_ + c12_temperature_coeff_ * (tlocal - reference_temperature_);

        RealType const
        c44 = c44_ + c44_temperature_coeff_ * (tlocal - reference_temperature_);

        CP::computeCubicElasticityTensor(c11, c12, c44, C_unrotated_);

        if (verbosity_ > 2) {
          std::cout << "tlocal: " << tlocal << std::endl;
          std::cout << "c11, c12, c44: " << c11 << c12 << c44 << std::endl;
        }
      }

      if (read_orientations_from_mesh_) {
        Teuchos::ArrayRCP<double*> const &
        rotation_matrix_transpose = workset.wsLatticeOrientation;

        orientation_matrix(0,0) = rotation_matrix_transpose[cell][0];
        orientation_matrix(0,1) = rotation_matrix_transpose[cell][1];
        orientation_matrix(0,2) = rotation_matrix_transpose[cell][2];
        orientation_matrix(1,0) = rotation_matrix_transpose[cell][3];
        orientation_matrix(1,1) = rotation_matrix_transpose[cell][4];
        orientation_matrix(1,2) = rotation_matrix_transpose[cell][5];
        orientation_matrix(2,0) = rotation_matrix_transpose[cell][6];
        orientation_matrix(2,1) = rotation_matrix_transpose[cell][7];
        orientation_matrix(2,2) = rotation_matrix_transpose[cell][8];
      }
      else {
        orientation_matrix = element_block_orientation_;
      }

      // Set the rotated elasticity tensor, slip normals, slip directions, 
      // and projection operator
      C_ = Intrepid2::kronecker(orientation_matrix, C_unrotated_);
      for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
        slip_systems_.at(num_ss).s_ = orientation_matrix * s_unrotated_[num_ss];
        slip_systems_.at(num_ss).n_ = orientation_matrix * n_unrotated_[num_ss];
        slip_systems_.at(num_ss).projector_ =
          Intrepid2::dyad(slip_systems_.at(num_ss).s_, slip_systems_.at(num_ss).n_);
      }

			// TODO: Try to preprocess this
			// Currently this possibly has a dependency on s_ and n_
  		for (int sf_index(0); sf_index < num_family_; ++sf_index) {
    		auto &
    		slip_family = slip_families_[sf_index];

    		slip_family.phardening_parameters_->createLatentMatrix(slip_family, slip_systems_); 

        if (verbosity_ > 2) {
          std::cout << slip_family.latent_matrix_ << std::endl;
        }
			}

      equivalent_plastic_strain = 
        Sacado::ScalarValue<ScalarT>::eval(eqps(cell, pt));

      // Copy data from Albany fields into local data structures
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          F_np1(i, j) = def_grad(cell, pt, i, j);
          Fp_n(i, j) = previous_plastic_deformation(cell, pt, i, j);
        }
      }

      for (int s(0); s < num_slip_; ++s) {
        slip_n[s] = (*(previous_slips[s]))(cell, pt);
        //
        // initialize state n+1 with either (a) zero slip increment or (b) a 
        // predictor
        //
        slip_np1[s] = slip_n[s];
        if (apply_slip_predictor_ == true) {
          slip_dot_n[s] = (*(rate_slip_n[s]))(cell, pt);
          slip_np1[s] += dt * slip_dot_n[s];
        }
        state_hardening_n[s] = (*(state_hardening[s]))(cell, pt);
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


      Minimizer minz;
      minz.rel_tol = minimizer_.rel_tol;
      minz.abs_tol = minimizer_.abs_tol;
      minz.max_num_iter = minimizer_.max_num_iter;
      minz.min_num_iter = minimizer_.min_num_iter;



      switch (integration_scheme_) {

        default:
        break;
        
				case CP::IntegrationScheme::EXPLICIT:
        {
          CP::PlasticityState<ScalarT, CP::MAX_DIM> plasticity_state(num_dims_, Fp_n);
          // TODO: is this the right rate?
          CP::SlipState<ScalarT, CP::MAX_SLIP> slip_state(num_slip_, state_hardening_n,
              slip_n, slip_dot_n);

          CP::Integrator<EvalT, CP::MAX_DIM, CP::MAX_SLIP> *
          integrator = new CP::ExplicitIntegrator<EvalT, CP::MAX_DIM, CP::MAX_SLIP>(
                slip_systems_,
                slip_families_,
                plasticity_state,
                slip_state,
                C_,
                F_np1,
                dt
              );

          Intrepid2::Vector<ScalarT, CP::MAX_SLIP>
          residual(num_slip_);

          integrator->update(residual);

          norm_slip_residual = 
            Sacado::ScalarValue<ScalarT>::eval(norm(residual));

          Fp_np1 = plasticity_state.Fp_np1_;
          Lp_np1 = plasticity_state.Lp_np1_;
          sigma_np1 = plasticity_state.sigma_np1_;
          S_np1 = plasticity_state.S_np1_;

          state_hardening_np1 = slip_state.hardening_np1_;
          slip_resistance = slip_state.resistance_;
          slip_np1 = slip_state.slip_np1_;
          shear_np1 = slip_state.shear_np1_;
#if 0
          for(int i = 0; i < num_slip_; ++i) {
            // initial guess for x is slip_n + dt * rate_slip_n
            x(i) = slip_n[i] + dt * (*(rate_slip_n[i]))(cell, pt);
          }

          using NLS =
          CP::ExplicitUpdateNLS<CP::MAX_DIM, CP::MAX_SLIP, EvalT>;

          NLS
          explicit_nls(
              C_,
              slip_systems_,
              slip_families_,
              Fp_n,
              state_hardening_n,
              slip_n,
              F_np1,
              dt);

          using STEP = Intrepid2::StepBase<NLS, ValueT, CP::NLS_DIM>;

          std::unique_ptr<STEP>
          pstep = Intrepid2::stepFactory<NLS, ValueT, CP::NLS_DIM>(step_type_);

          LCM::MiniSolver<Minimizer, STEP, NLS, EvalT, CP::NLS_DIM>
          mini_solver(minimizer_, *pstep, explicit_nls, x);

          for(int i = 0; i < num_slip_; ++i) {
            slip_np1[i] = x[i];
          }

          // Compute Lp_np1, and Fp_np1
          CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
            slip_systems_, 
            dt,
            slip_n, 
            slip_np1, 
            Fp_n, 
            Lp_np1, 
            Fp_np1);

          // Compute sigma_np1, S_np1, and shear_np1
          CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, ScalarT, ScalarT>(
            slip_systems_, 
            C_, 
            F_np1, 
            Fp_np1, 
            sigma_np1, 
            S_np1, 
            shear_np1);

          if (dt > 0.0) {
            rate_slip = (slip_np1 - slip_n) / dt;
          }
          else {
            rate_slip.fill(Intrepid2::ZEROS);
          }

          // Compute state_hardening_np1
          CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
            slip_systems_, 
            slip_families_,
            dt,
            rate_slip, 
            state_hardening_n, 
            state_hardening_np1,
            slip_resistance);
#endif
        } // end case CP::IntegrationScheme::EXPLICIT:
        break;
        
				case CP::IntegrationScheme::IMPLICIT:
        {

          //
          // Choose residual type
          //
          switch (residual_type_)
          {
						case CP::ResidualType::SLIP:
            {
              using NLS = CP::ResidualSlipNLS<CP::MAX_DIM, CP::MAX_SLIP, EvalT>;

              NLS
              slip_nls(
                  C_,
                  slip_systems_,
                  slip_families_,
                  Fp_n,
                  state_hardening_n,
                  slip_n,
                  F_np1,
                  dt);

              for(int i = 0; i < num_slip_; ++i) {
                // initial guess for x is slip_np1 (predictor, see above)
                x(i) = Sacado::ScalarValue<ScalarT>::eval(slip_np1(i));
              }



              using STEP = Intrepid2::StepBase<NLS, ValueT, CP::NLS_DIM>;

              std::unique_ptr<STEP>
              pstep = 
                Intrepid2::stepFactory<NLS, ValueT, CP::NLS_DIM>(step_type_);

              LCM::MiniSolver<Minimizer, STEP, NLS, EvalT, CP::NLS_DIM>
              mini_solver(minz, *pstep, slip_nls, x);

              for(int i=0; i<num_slip_; ++i) {
                slip_np1[i] = x[i];
              }

              if(dt > 0.0){
                rate_slip = (slip_np1 - slip_n) / dt;
              }
              else{
                rate_slip.fill(Intrepid2::ZEROS);
              }

              // Compute state_hardening_np1
              CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
                slip_systems_, 
                slip_families_,
                dt,
                rate_slip, 
                state_hardening_n, 
                state_hardening_np1,
                slip_resistance);
            }
            break;

						case CP::ResidualType::SLIP_HARDNESS:
            {
              using NLS =
                  CP::ResidualSlipHardnessNLS<CP::MAX_DIM, CP::MAX_SLIP, EvalT>;

              NLS
              slip_state_hardening_nls(
                  C_,
                  slip_systems_,
                  slip_families_,
                  Fp_n,
                  state_hardening_n,
                  slip_n,
                  F_np1,
                  dt);

              for(int i=0; i<num_slip_; ++i) {
                // initial guess for x(0:num_slip_-1) from predictor
                x(i) = Sacado::ScalarValue<ScalarT>::eval(slip_np1(i));
                // initial guess for x(num_slip_:2*num_slip_) from predictor
                x(i + num_slip_) = 
                  Sacado::ScalarValue<ScalarT>::eval(state_hardening_n(i));
              }

              using STEP = Intrepid2::StepBase<NLS, ValueT, CP::NLS_DIM>;

              std::unique_ptr<STEP>
              pstep = Intrepid2::stepFactory<NLS, ValueT, CP::NLS_DIM>(step_type_);

              LCM::MiniSolver<Minimizer, STEP, NLS, EvalT, CP::NLS_DIM>
              mini_solver(minz, *pstep, slip_state_hardening_nls, x);

              for(int i=0; i<num_slip_; ++i) {
                slip_np1[i] = x[i];
                state_hardening_np1[i] = x[i + num_slip_];
              }

              if(dt > 0.0){
                rate_slip = (slip_np1 - slip_n) / dt;
              }
              else{
                rate_slip.fill(Intrepid2::ZEROS);
              }

              // Compute state_hardening_np1
              CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
                slip_systems_, 
                slip_families_,
                dt,
                rate_slip, 
                state_hardening_n, 
                state_hardening_np1,
                slip_resistance);
            }
            break;

            default:
              std::cerr << "You must supply a residual type." << std::endl;
              exit(1);
            break;
          }

          if(!minz.converged){
            if(verbosity_ > 2){
              std::cout << "\n**** CrystalPlasticityModel computeState()";
              std::cout << " failed to converge.\n" << std::endl;
              minz.printReport(std::cout);
            }
            update_state_successful = false;
            forceGlobalLoadStepReduction();
          }

          // cases in which the model is subject to divergence
          // more work to do in the nonlinear systems
          if(minz.failed){
            if (verbosity_ > 2){
              std::cout << "\n**** CrystalPlasticityModel computeState() ";
              std::cout << "exited due to failure criteria.\n" << std::endl;
            }
            update_state_successful = false;
            forceGlobalLoadStepReduction();
          }

          // We now have the solution for slip_np1, including sensitivities 
          // (if any). Re-evaluate all the other state variables based on 
          // slip_np1.

          // Compute Lp_np1, and Fp_np1
          CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP, ScalarT>(
            slip_systems_, 
            dt,
            slip_n, 
            slip_np1, 
            Fp_n, 
            Lp_np1, 
            Fp_np1);

          // Compute sigma_np1, S_np1, and shear_np1
          CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP, ScalarT, ScalarT>(
            slip_systems_, 
            C_, 
            F_np1, 
            Fp_np1, 
            sigma_np1, 
            S_np1, 
            shear_np1);
        } // end case CP::IntegrationScheme::IMPLICIT:
        break;
      } // end switch (integration_scheme_)

      // Compute the residual norm 
      norm_slip_residual = std::sqrt(2.0 * minz.final_value);
      residual_iter = minz.num_iter;

      if(update_state_successful){

        // Compute the equivalent plastic strain from the velocity gradient:
        //  eqps_dot = sqrt[2/3* sym(Lp) : sym(Lp)]
        Intrepid2::Tensor<ScalarT, CP::MAX_DIM> const
        Dp = Intrepid2::sym(Lp_np1);

        RealType const
        delta_eqps = dt * std::sqrt(2.0 / 3.0 *
          Sacado::ScalarValue<ScalarT>::eval(Intrepid2::dotdot(Dp,Dp)));

        equivalent_plastic_strain += delta_eqps;

        eqps(cell, pt) = equivalent_plastic_strain;

        // The xtal rotation from the polar decomp of Fe.
        Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
        Fe(num_dims_);

        Intrepid2::Tensor<ScalarT, CP::MAX_DIM>
        Re_np1(num_dims_);

        // Multiplicatively decompose F to get Fe
        Fe = F_np1 * Intrepid2::inverse(Fp_np1);

        // Compute polar rotation to get Re
        Re_np1 = Intrepid2::polar_rotation(Fe);

        ///
        /// Copy data from local data structures back into Albany fields
        ///

        // mechanical heat source
        if (have_temperature_) {
          source(cell, pt) = 0.0;
          if (dt > 0.0) {

            rate_slip = (slip_np1 - slip_n) / dt;

            RealType
            plastic_dissipation(0.0);

            for (int slip_system(0); slip_system < num_slip_; ++slip_system) {
              plastic_dissipation += Sacado::ScalarValue<ScalarT>::eval(
                rate_slip[slip_system] * shear_np1[slip_system]);
            }
            source(cell, pt) = 0.9 / (density_ * heat_capacity_) * plastic_dissipation;
          }
        }

        // residual norm
        cp_residual(cell, pt) = norm_slip_residual;
        cp_residual_iter(cell,pt) = residual_iter;

        // num_dims_ x num_dims_ dimensional array variables
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            xtal_rotation(cell, pt, i, j) = Re_np1(i, j);
            plastic_deformation(cell, pt, i, j) = Fp_np1(i, j);
            stress(cell, pt, i, j) = sigma_np1(i, j);
            velocity_gradient(cell, pt, i, j) = Lp_np1(i, j);
          }
        }

        // num_slip_ dimensional array variables
        for (int s(0); s < num_slip_; ++s) {
          (*(slips[s]))(cell, pt) = slip_np1[s];
          (*(hards[s]))(cell, pt) = state_hardening_np1[s];
          (*(shears[s]))(cell, pt) = shear_np1[s];
          // storing the slip rate for the predictor
          if (dt > 0.0) {
            (*(rate_slip_np1[s]))(cell, pt) = (slip_np1[s] - slip_n[s]) / dt;
          }
          else {
            (*(rate_slip_np1[s]))(cell, pt) = 0.0;
          }
        }

        if(write_data_file_) {
          if (cell == 0 && pt == 0) {

            std::ofstream
            data_file("output.dat", std::fstream::app);

            Intrepid2::Tensor<RealType, CP::MAX_DIM>
            P(num_dims_);

            data_file << "\n" << "time: ";
            data_file << std::setprecision(12);
            data_file << Sacado::ScalarValue<ScalarT>::eval(time(0));
            data_file << "     dt: ";
            data_file << std::setprecision(12) << dt << " \n";

            for (int s(0); s < num_slip_; ++s) {
              data_file << "\n" << "P" << s << ": ";
              P = slip_systems_.at(s).projector_;
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
      } // end update_state_successful
    } // end loop over integration points
  } // end loop over elements
} // computeState

} // namespace LCM
