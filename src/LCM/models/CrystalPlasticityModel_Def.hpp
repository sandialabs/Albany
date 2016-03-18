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

namespace LCM
{

template<typename EvalT, typename Traits>
CrystalPlasticityModel<EvalT, Traits>::
CrystalPlasticityModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    num_slip_(p->get<int>("Number of Slip Systems", 0))
{
  // Store an RCP to the NOX status test, if available
  if (p->isParameter("NOX Status Test")) {
    nox_status_test_ = p->get< Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> >("NOX Status Test");
  }

  integration_scheme_ = IntegrationScheme::EXPLICIT;
  if (p->isParameter("Integration Scheme")) {
    std::string integration_scheme_string = p->get<std::string>(
        "Integration Scheme");
    if (integration_scheme_string == "Implicit") {
      integration_scheme_ = IntegrationScheme::IMPLICIT;
    }
    else if (integration_scheme_string == "Explicit") {
      integration_scheme_ = IntegrationScheme::EXPLICIT;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "\n**** Error in CrystalPlasticityModel, invalid value for \
          \"Integration Scheme\", must be \
          \"Implicit\" or \
          \"Explicit\".\n");
    }
  }

  residual_type_ = ResidualType::SLIP;
  if (p->isParameter("Residual Type")) {
    std::string residual_type_string = p->get<std::string>(
        "Residual Type");
    if (residual_type_string == "Slip") {
      residual_type_ = ResidualType::SLIP;
    }
    else if (residual_type_string == "Slip Hardness") {
      residual_type_ = ResidualType::SLIP_HARDNESS;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "\n**** Error in CrystalPlasticityModel, invalid value for \
            \"Residual Type\", must be \
            \"Slip\" or \
            \"Slip Hardness\".\n");
    }
  }

  step_type_ = Intrepid2::StepType::NEWTON;
  if (p->isParameter("Nonlinear Solver Step Type")) {
    std::string step_type_string = p->get<std::string>(
        "Nonlinear Solver Step Type");
    if (step_type_string == "Newton") {
      step_type_ = Intrepid2::StepType::NEWTON;
    }
    else if (step_type_string == "Trust Region") {
      step_type_ = Intrepid2::StepType::TRUST_REGION;
    }
    else if (step_type_string == "Conjugate Gradient") {
      step_type_ = Intrepid2::StepType::CG;
    }
    else if (step_type_string == "Line Search Regularized") {
      step_type_ = Intrepid2::StepType::LINE_SEARCH_REG;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "\n**** Error in CrystalPlasticityModel, invalid value for \
            \"Nonlinear Solver Step Type\", must be \
            \"Newton\", \
            \"Trust Region\", \
            \"Conjugate Gradient\", or \
            \"Line Search Regularized\".\n");
    }
  }

  implicit_nonlinear_solver_relative_tolerance_ = p->get<double>(
      "Implicit Integration Relative Tolerance",
      1.0e-6);

  implicit_nonlinear_solver_absolute_tolerance_ = p->get<double>(
      "Implicit Integration Absolute Tolerance",
      1.0e-10);

  implicit_nonlinear_solver_max_iterations_ = p->get<int>(
      "Implicit Integration Max Iterations",
      100);

  implicit_nonlinear_solver_min_iterations_ = p->get<int>(
      "Implicit Integration Min Iterations",
      2);

  apply_slip_predictor_ = p->get<bool>(
      "Apply Slip Predictor",
      true);

  verbosity_ = p->get<int>(
      "Verbosity",
      0);

  write_data_file_ = p->get<bool>(
      "Write Data File",
      false);

  slip_systems_.resize(num_slip_);

  if (verbosity_ > 2) {
    std::cout << ">>> in cp constructor\n";
    std::cout << ">>> parameter list:\n" << *p << std::endl;
  }

  //
  // Obtain crystal elasticity constants and populate elasticity tensor
  //
  Teuchos::ParameterList e_list = p->sublist("Crystal Elasticity");
  // assuming cubic symmetry

  c11_ = e_list.get<RealType>("C11");
  c12_ = e_list.get<RealType>("C12");
  c44_ = e_list.get<RealType>("C44");
  c11_temperature_coeff_ = e_list.get<RealType>("M11",0.0);
  c12_temperature_coeff_ = e_list.get<RealType>("M12",0.0);
  c44_temperature_coeff_ = e_list.get<RealType>("M44",0.0);
  reference_temperature_ = e_list.get<RealType>("Reference Temperature",0.0);


  Intrepid2::Tensor4<ScalarT, CP::MAX_DIM> C(Intrepid2::ZEROS);
  C.set_dimension(num_dims_);
  CP::computeCubicElasticityTensor(c11_, c12_, c44_, C);

  if (verbosity_ > 2) {
    // print elasticity tensor
    std::cout << ">>> C :" << std::endl << C << std::endl;
  }

  // TODO check if basis is given else default
  // NOTE default to coordinate axes; construct 3rd direction if only 2 given
  for (int i = 0; i < num_dims_; ++i) {
    std::vector<RealType> b_temp = e_list.get<Teuchos::Array<RealType>>(
        Albany::strint("Basis Vector", i + 1)).toVector();
    RealType norm = 0.;
    for (int j = 0; j < num_dims_; ++j) {
      norm += b_temp[j] * b_temp[j];
    }
    RealType const inverse_norm = 1. / std::sqrt(norm);

    // TODO check zero, rh system
    // Filling columns of transformation with basis vectors
    // We are forming R^{T} which is equivalent to the direction cosine matrix
    orientation_.set_dimension(num_dims_);
    for (int j = 0; j < num_dims_; ++j) {
      orientation_(j, i) = b_temp[j] * inverse_norm;
    }
  }

  if (verbosity_ > 2) {
    // print rotation tensor employed for transformations
    std::cout << ">>> orientation_ :\n" << orientation_ << std::endl;
  }

  //
  // rotate elastic tensor and slip systems to match given orientation
  //
  // Intrepid2::Tensor4<ScalarT, CP::MAX_DIM> C_;
  C_ = Intrepid2::kronecker(orientation_, C);

  if (verbosity_ > 2) {
    // print rotated elasticity tensor
    std::cout << ">>> C_ :" << std::endl << C_ << std::endl;
  }

  // Get slip system information
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    Teuchos::ParameterList ss_list = 
      p->sublist(Albany::strint("Slip System", num_ss + 1));

    //
    // Obtain and normalize slip directions. Miller indices need to be normalized.
    //
    std::vector<RealType> s_temp = ss_list.get<Teuchos::Array<RealType>>(
        "Slip Direction").toVector();
    Intrepid2::Vector<RealType, CP::MAX_DIM> s_temp_normalized(num_dims_);
    for (int i = 0; i < num_dims_; ++i) {
      s_temp_normalized[i] = s_temp[i];
    }
    s_temp_normalized = Intrepid2::unit(s_temp_normalized);
    slip_systems_[num_ss].s_.set_dimension(num_dims_);
    slip_systems_[num_ss].s_ = orientation_ * s_temp_normalized;

    //
    // Obtain and normalize slip normals. Miller indices need to be normalized.
    //
    std::vector<RealType> n_temp = ss_list.get<Teuchos::Array<RealType>>(
        "Slip Normal").toVector();
    Intrepid2::Vector<RealType, CP::MAX_DIM> n_temp_normalized(num_dims_);
    for (int i = 0; i < num_dims_; ++i) {
      n_temp_normalized[i] = n_temp[i];
    }
    n_temp_normalized = Intrepid2::unit(n_temp_normalized);
    slip_systems_[num_ss].n_.set_dimension(num_dims_);
    slip_systems_[num_ss].n_ = orientation_ * n_temp_normalized;

    // Print each slip direction and slip normal after transformation
    if (verbosity_ > 2) {
      std::cout << ">>> slip direction " << num_ss + 1 << ": "
          << slip_systems_[num_ss].s_ << std::endl;
      std::cout << ">>> slip normal " << num_ss + 1 << ": "
          << slip_systems_[num_ss].n_ << std::endl;
    }

    slip_systems_[num_ss].projector_.set_dimension(num_dims_);
    slip_systems_[num_ss].projector_ = Intrepid2::dyad(
        slip_systems_[num_ss].s_,
        slip_systems_[num_ss].n_);

    // Print projector
    if (verbosity_ > 2) {
      std::cout << ">>> projector_ " << num_ss + 1 << ": "
          << slip_systems_[num_ss].projector_ << std::endl;
    }

    //
    // Obtain flow rule parameters
    //
    std::string name_flow_rule = ss_list.get<std::string>("Flow Rule");
    Teuchos::ParameterList f_list = p->sublist(name_flow_rule);
    std::string type_flow_rule = f_list.get<std::string>("Type");

    if (type_flow_rule == "Power Law") {
      slip_systems_[num_ss].flow_rule = CP::FlowRule::POWER_LAW;
      slip_systems_[num_ss].rate_slip_reference_ = 
        f_list.get<RealType>("Gamma Dot", 0.0);
      slip_systems_[num_ss].exponent_rate_ = 
        f_list.get<RealType>("Gamma Exponent", 0.0);
    }
    else if (type_flow_rule == "Thermal Activation") {
      slip_systems_[num_ss].flow_rule = CP::FlowRule::THERMAL_ACTIVATION;
      slip_systems_[num_ss].rate_slip_reference_ = 
        f_list.get<RealType>("Gamma Dot", 0.0);
      slip_systems_[num_ss].energy_activation_ = 
        f_list.get<RealType>("Activation Energy", 0.0);
    }

    //
    // Obtain hardening law parameters
    //
    std::string name_hardening_law = ss_list.get<std::string>("Hardening Law");
    Teuchos::ParameterList h_list = p->sublist(name_hardening_law);
    std::string type_hardening_law = h_list.get<std::string>("Type");

    if (type_hardening_law == "Exponential") {
      slip_systems_[num_ss].hardening_law = CP::HardeningLaw::EXPONENTIAL;
      slip_systems_[num_ss].H_ = h_list.get<RealType>("Hardening", 0.0);
      slip_systems_[num_ss].Rd_ = 
        h_list.get<RealType>("Hardening Exponent", 0.0);
      slip_systems_[num_ss].tau_critical_ = 
        h_list.get<RealType>("Tau Critical", 0.0);
    }
    else if (type_hardening_law == "Saturation") {
      slip_systems_[num_ss].hardening_law = CP::HardeningLaw::SATURATION;
      slip_systems_[num_ss].resistance_slip_initial_ = 
        h_list.get<RealType>("Initial Slip Resistance", 0.0);
      // temporary workaround
      slip_systems_[num_ss].tau_critical_ = 
        slip_systems_[num_ss].resistance_slip_initial_;
      slip_systems_[num_ss].rate_hardening_ = 
        h_list.get<RealType>("Hardening Rate", 0.0);
      slip_systems_[num_ss].stress_saturation_initial_ = 
        h_list.get<RealType>("Initial Saturation Stress", 0.0);
      slip_systems_[num_ss].exponent_saturation_ = 
        h_list.get<RealType>("Saturation Exponent", 0.0);
    }
    else {
      slip_systems_[num_ss].hardening_law = CP::HardeningLaw::UNDEFINED;
    }

    if (verbosity_ > 2) {
      std::cout << "Slip system number " << num_ss << std::endl;
      std::cout << "Hardening law ";
      std::cout << static_cast<int>(slip_systems_[num_ss].hardening_law);
      std::cout << std::endl;
      std::cout << "H " << slip_systems_[num_ss].H_ << std::endl;
      std::cout << "Rd " << slip_systems_[num_ss].Rd_ << std::endl;
      std::cout << "Tau critical ";
      std::cout << slip_systems_[num_ss].tau_critical_ << std::endl;
      std::cout << "Initial slip resistance ";
      std::cout << slip_systems_[num_ss].resistance_slip_initial_ << std::endl;
      std::cout << "Hardening rate ";
      std::cout << slip_systems_[num_ss].rate_hardening_ << std::endl;
      std::cout << "Initial saturation stress ";
      std::cout << slip_systems_[num_ss].stress_saturation_initial_ << std::endl;
      std::cout << "Saturation exponent ";
      std::cout << slip_systems_[num_ss].exponent_saturation_ << std::endl;
    }

  }

  //
  // retrive appropriate field name strings (ref to problems/FieldNameMap)
  //
  std::string const eqps_string = (*field_name_map_)["eqps"];
  std::string const Re_string = (*field_name_map_)["Re"];
  std::string const cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string const Fp_string = (*field_name_map_)["Fp"];
  std::string const L_string = (*field_name_map_)["Velocity_Gradient"];
  std::string const F_string = (*field_name_map_)["F"];
  std::string const J_string = (*field_name_map_)["J"];
  std::string const source_string = (*field_name_map_)["Mechanical_Source"];
  std::string const residual_string = (*field_name_map_)["CP_Residual"];

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
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output eqps", false));

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
    std::string const g = Albany::strint("gamma", num_ss + 1, '_');
    std::string const gamma_string = (*field_name_map_)[g];
    std::string const output_gamma_string = "Output " + gamma_string;
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
    std::string const g_dot = Albany::strint("gamma_dot", num_ss + 1, '_');
    std::string const gamma_dot_string = (*field_name_map_)[g_dot];
    std::string const output_gamma_dot_string = "Output " + gamma_dot_string;
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
    std::string const t_h = Albany::strint("tau_hard", num_ss + 1, '_');
    std::string const tau_hard_string = (*field_name_map_)[t_h];
    std::string const output_tau_hard_string = "Output " + tau_hard_string;
    this->eval_field_map_.insert(
        std::make_pair(tau_hard_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(tau_hard_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_tau_hard_string, false));
  }

  // taus - output resolved shear stress for debugging - not stated
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string const t = Albany::strint("tau", num_ss + 1, '_');
    std::string const tau_string = (*field_name_map_)[t];
    std::string const output_tau_string = "Output " + tau_string;
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
}

//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>>dep_fields,
std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
{

  if(verbosity_ > 2) {
    std::cout << ">>> in cp compute state\n";
  }

  // retrive appropriate field name strings
  std::string const eqps_string = (*field_name_map_)["eqps"];
  std::string const Re_string = (*field_name_map_)["Re"];
  std::string const cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string const Fp_string = (*field_name_map_)["Fp"];
  std::string const L_string = (*field_name_map_)["Velocity_Gradient"];
  std::string const residual_string = (*field_name_map_)["CP_Residual"];
  std::string const source_string = (*field_name_map_)["Mechanical_Source"];
  std::string const F_string = (*field_name_map_)["F"];
  std::string const J_string = (*field_name_map_)["J"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad = *dep_fields[F_string];
  PHX::MDField<ScalarT> J = *dep_fields[J_string];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> eqps = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> xtal_rotation = *eval_fields[Re_string];
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> plastic_deformation = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> velocity_gradient = *eval_fields[L_string];
  PHX::MDField<ScalarT> source;
  if (have_temperature_) {
    source = *eval_fields[source_string];
  }
  PHX::MDField<ScalarT> cp_residual = *eval_fields[residual_string];

  PHX::MDField<ScalarT> time = *eval_fields["Time"];
  // extract slip on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>> slips;
  std::vector<Albany::MDArray *> previous_slips;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string const g = Albany::strint("gamma", num_ss + 1, '_');
    std::string const gamma_string = (*field_name_map_)[g];
    slips.push_back(eval_fields[gamma_string]);
    previous_slips.push_back(
    &((*workset.stateArrayPtr)[gamma_string + "_old"]));
  }
  // extract slip rate on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>> slips_dot;
  std::vector<Albany::MDArray *> previous_slips_dot;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string const g_dot = Albany::strint("gamma_dot", num_ss + 1, '_');
    std::string const gamma_dot_string = (*field_name_map_)[g_dot];
    slips_dot.push_back(eval_fields[gamma_dot_string]);
    previous_slips_dot.push_back(
    &((*workset.stateArrayPtr)[gamma_dot_string + "_old"]));
  }
  // extract hardening on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>> hards;
  std::vector<Albany::MDArray *> previous_hards;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string const t_h = Albany::strint("tau_hard", num_ss + 1, '_');
    std::string const tau_hard_string = (*field_name_map_)[t_h];
    hards.push_back(eval_fields[tau_hard_string]);
    previous_hards.push_back(
    &((*workset.stateArrayPtr)[tau_hard_string + "_old"]));
  }
  // store shear on each slip system for output
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT>>> shears;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string const t = Albany::strint("tau", num_ss + 1, '_');
    std::string const tau_string = (*field_name_map_)[t];
    shears.push_back(eval_fields[tau_string]);
  }

  // get state variables
  Albany::MDArray previous_plastic_deformation =
  (*workset.stateArrayPtr)[Fp_string + "_old"];
  ScalarT tau, gamma, dgamma;
  RealType dt = Sacado::ScalarValue<ScalarT>::eval(delta_time(0));
  RealType tcurrent = Sacado::ScalarValue<ScalarT>::eval(time(0));

  // -- Local variables for implicit integration routine --

  // Known quantities
  Intrepid2::Tensor<RealType, CP::MAX_DIM> Fp_n(num_dims_);
  Intrepid2::Vector<RealType, CP::MAX_SLIP> slip_n(num_slip_);
  Intrepid2::Vector<RealType, CP::MAX_SLIP> slip_dot_n(num_slip_);
  Intrepid2::Vector<RealType, CP::MAX_SLIP> hardness_n(num_slip_);
  Intrepid2::Tensor<ScalarT, CP::MAX_DIM> F_np1(num_dims_);

  // Unknown quantities
  Intrepid2::Tensor<ScalarT, CP::MAX_DIM> Lp_np1(num_dims_);
  Intrepid2::Tensor<ScalarT, CP::MAX_DIM> Fp_np1(num_dims_);
  Intrepid2::Tensor<ScalarT, CP::MAX_DIM> sigma_np1(num_dims_);
  Intrepid2::Tensor<ScalarT, CP::MAX_DIM> S_np1(num_dims_);
  Intrepid2::Vector<ScalarT, CP::MAX_SLIP> rate_slip(num_slip_);
  Intrepid2::Vector<ScalarT, CP::MAX_SLIP> slip_np1(num_slip_);
  Intrepid2::Vector<ScalarT, CP::MAX_SLIP> shear_np1(num_slip_);
  Intrepid2::Vector<ScalarT, CP::MAX_SLIP> hardness_np1(num_slip_);
  Intrepid2::Vector<ScalarT, CP::MAX_SLIP> slip_computed(num_slip_);

  RealType norm_slip_residual;
  RealType equivalent_plastic_strain;

  bool update_state_successful = true;

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {

      if (have_temperature_) {

        RealType const tlocal = 
          Sacado::ScalarValue<ScalarT>::eval(temperature_(cell,pt));

        RealType const c11 = c11_ + c11_temperature_coeff_ *
          (tlocal - reference_temperature_);
        RealType const c12 = c12_ + c12_temperature_coeff_ *
          (tlocal - reference_temperature_);
        RealType const c44 = c44_ + c44_temperature_coeff_ *
          (tlocal - reference_temperature_);
        Intrepid2::Tensor4<ScalarT, CP::MAX_DIM> C(num_dims_);
        CP::computeCubicElasticityTensor(c11, c12, c44, C);
        C_ = Intrepid2::kronecker(orientation_, C);

        if (verbosity_ > 2) {
          std::cout << "tlocal: " << tlocal << std::endl;
          std::cout << "c11, c12, c44: " << c11 << c12 << c44 << std::endl;
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
          slip_dot_n[s] = (*(previous_slips_dot[s]))(cell, pt);
          slip_np1[s] += dt * slip_dot_n[s];
        }

        hardness_n[s] = (*(previous_hards[s]))(cell, pt);
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

      switch (integration_scheme_) {

        default:
        break;
        
        case IntegrationScheme::EXPLICIT:
        {
          // compute sigma_np1, S_np1, and shear_np1 using Fp_n
          CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_, 
            C_, 
            F_np1, 
            Fp_n, 
            sigma_np1, 
            S_np1, 
            shear_np1);

          for (int iSlipSystem(0); iSlipSystem < num_slip_; ++iSlipSystem) {
            rate_slip[iSlipSystem] = 
              (*(previous_slips_dot[iSlipSystem]))(cell, pt);
          }

          // compute hardness_np1 using slip_n
          CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_, 
            dt,
            rate_slip, 
            hardness_n, 
            hardness_np1);

          // compute slip_np1
          CP::updateSlip<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_,
            dt,
            hardness_np1,
            shear_np1,
            slip_n,
            slip_np1);

          // compute Lp_np1, and Fp_np1
          CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_, 
            dt,
            slip_n, 
            slip_np1, 
            Fp_n, 
            Lp_np1, 
            Fp_np1);

          // compute sigma_np1, S_np1, and shear_np1 using Fp_np1
          CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_, 
            C_, 
            F_np1, 
            Fp_np1, 
            sigma_np1, 
            S_np1, 
            shear_np1);

          // compute slip_np1
          CP::updateSlip<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_,
            dt,
            hardness_np1,
            shear_np1,
            slip_n,
            slip_computed);

          norm_slip_residual = 
            Sacado::ScalarValue<ScalarT>::eval(norm(slip_np1 - slip_computed));

          if(verbosity_ > 2) {
            std::cout << "CP model explicit integration residual ";
            std::cout << norm_slip_residual << std::endl;
          }
        }

        break;
        
        case IntegrationScheme::IMPLICIT:
        {

          constexpr
          Intrepid2::Index
          NLS_DIM = 2 * CP::MAX_SLIP;

          using ValueT = typename Sacado::ValueType<ScalarT>::type;
          using MIN = Intrepid2::Minimizer<ValueT, NLS_DIM>;

          MIN
          minimizer;

          // unknowns array
          Intrepid2::Vector<ScalarT, 2 * CP::MAX_SLIP> x;

          minimizer.rel_tol = implicit_nonlinear_solver_relative_tolerance_;
          minimizer.abs_tol = implicit_nonlinear_solver_absolute_tolerance_;
          minimizer.max_num_iter = implicit_nonlinear_solver_max_iterations_;
          minimizer.min_num_iter = implicit_nonlinear_solver_min_iterations_;

          //
          // Chose residual type
          //
          switch (residual_type_)
          {
            
            case ResidualType::SLIP:
            {
              using NLS =
                  CP::CrystalPlasticityNLS<CP::MAX_DIM, CP::MAX_SLIP, EvalT>;

              NLS
              slip_nls(
                  C_,
                  slip_systems_,
                  Fp_n,
                  hardness_n,
                  slip_n,
                  F_np1,
                  dt);

              using STEP = Intrepid2::StepBase<NLS, ValueT, NLS_DIM>;

              std::unique_ptr<STEP>
              pstep =
                  Intrepid2::stepFactory<NLS, ValueT, NLS_DIM>(step_type_);

              STEP &
              step = *pstep;

              // unknowns, which are slip_np1
              x.set_dimension(num_slip_);

              for(int i=0; i<num_slip_; ++i) {
                // initial guess for x is slip_np1 (predictor, see above)
                x(i) = Sacado::ScalarValue<ScalarT>::eval(slip_np1(i));
              }

              LCM::MiniSolver<MIN, STEP, NLS, EvalT, NLS_DIM>
              mini_solver(minimizer, step, slip_nls, x);

              for(int i=0; i<num_slip_; ++i) {
                slip_np1[i] = x[i];
              }

              if(dt > 0.0){
                rate_slip = (slip_np1 - slip_n) / dt;
              }
              else{
                rate_slip.fill(Intrepid2::ZEROS);
              }

              // Compute hardness_np1
              CP::updateHardness<CP::MAX_DIM, CP::MAX_SLIP>(
                slip_systems_, 
                dt,
                rate_slip, 
                hardness_n, 
                hardness_np1);

            }
          break;

          case ResidualType::SLIP_HARDNESS:
          {
            using NLS =
                CP::ResidualSlipHardnessNLS<CP::MAX_DIM, CP::MAX_SLIP, EvalT>;

            NLS
            slip_hardness_nls(
                C_,
                slip_systems_,
                Fp_n,
                hardness_n,
                slip_n,
                F_np1,
                dt);

            using STEP = Intrepid2::StepBase<NLS, ValueT, NLS_DIM>;

            std::unique_ptr<STEP>
            pstep =
                Intrepid2::stepFactory<NLS, ValueT, NLS_DIM>(step_type_);

            STEP &
            step = *pstep;

            // unknowns, which are slip_np1 followed by hardness_np1
            x.set_dimension(2 * num_slip_);

            for(int i=0; i<num_slip_; ++i) {
              // initial guess for x(0:num_slip_-1) is slip_np1 (predictor, see above)
              x(i) = Sacado::ScalarValue<ScalarT>::eval(slip_np1(i));
              // initial guess for x(num_slip_:2*num_slip_) is slip_np1 (predictor, see above)
              x(i + num_slip_) = 
                Sacado::ScalarValue<ScalarT>::eval(hardness_np1(i));
            }

            LCM::MiniSolver<MIN, STEP, NLS, EvalT, NLS_DIM>
            mini_solver(minimizer, step, slip_hardness_nls, x);

            for(int i=0; i<num_slip_; ++i) {
              slip_np1[i] = x[i];
              hardness_np1[i] = x[i + num_slip_];
            }

          }
          break;

          default:
            std::cerr << "You must supply a residual type." << std::endl;
            exit(1);
          break;

	  }

          if(!minimizer.converged){
	          std::cout << "\n**** CrystalPlasticityModel computeState() failed to converge.\n" << std::endl;
            minimizer.printReport(std::cout);
	          update_state_successful = false;
	          forceGlobalLoadStepReduction();
          }

          // cases in which the model is subject to divergence - more work to do in the nonlinear systems
          if(minimizer.failed){
            std::cout << "\n**** CrystalPlasticityModel computeState() exited due to failure criteria.\n" << std::endl;
            update_state_successful = false;
            forceGlobalLoadStepReduction();
          }


	  // We now have the solution for slip_np1, including sensitivities (if any)
	  // Re-evaluate all the other state variables based on slip_np1

	  // Compute Lp_np1, and Fp_np1
	  CP::applySlipIncrement<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_, 
	    dt,
	    slip_n, 
	    slip_np1, 
	    Fp_n, 
	    Lp_np1, 
	    Fp_np1);

	  // Compute sigma_np1, S_np1, and shear_np1
	  CP::computeStress<CP::MAX_DIM, CP::MAX_SLIP>(
            slip_systems_, 
	    C_, 
	    F_np1, 
	    Fp_np1, 
	    sigma_np1, 
	    S_np1, 
	    shear_np1);

	  // Compute the residual norm 
	  norm_slip_residual = std::sqrt(2.0 * minimizer.final_value);

        }
        break;

      }

      if(update_state_successful){

	// Compute the equivalent plastic strain from the velocity gradient:
	//  eqps_dot = sqrt[2/3* sym(Lp) : sym(Lp)]
	Intrepid2::Tensor<ScalarT, CP::MAX_DIM> const Dp = Intrepid2::sym(Lp_np1);
	RealType const delta_eqps = dt * std::sqrt(2.0 / 3.0 *
          Sacado::ScalarValue<ScalarT>::eval(Intrepid2::dotdot(Dp,Dp)));
	equivalent_plastic_strain += delta_eqps;
	eqps(cell, pt) = equivalent_plastic_strain;

	// The xtal rotation from the polar decomp of Fe.
	Intrepid2::Tensor<ScalarT, CP::MAX_DIM> Fe(num_dims_);
	Intrepid2::Tensor<ScalarT, CP::MAX_DIM> Re_np1(num_dims_);

	// Multiplicatively decompose F to get Fe
	Fe = F_np1 * (Intrepid2::inverse(Fp_np1));

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
            RealType plastic_dissipation(0.0);
            for (int slip_system(0); slip_system < num_slip_; ++slip_system) {
              plastic_dissipation += Sacado::ScalarValue<ScalarT>::eval(
                rate_slip[slip_system] * shear_np1[slip_system]);
            }
            source(cell, pt) = 0.9 / (density_ * heat_capacity_) * plastic_dissipation;
	  }
	}

	// residual norm
	cp_residual(cell, pt) = norm_slip_residual;

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
	  (*(hards[s]))(cell, pt) = hardness_np1[s];
	  (*(shears[s]))(cell, pt) = shear_np1[s];
	  // storing the slip rate for the predictor
	  if (dt > 0) {
	    (*(slips_dot[s]))(cell, pt) = (slip_np1[s] - slip_n[s]) / dt;
	  }
	  else {
	    (*(slips_dot[s]))(cell, pt) = 0.0;
	  }
	}

	if(write_data_file_) {
	  if (cell == 0 && pt == 0) {
	    std::ofstream data_file("output.dat", std::fstream::app);
	    Intrepid2::Tensor<RealType, CP::MAX_DIM> P(num_dims_);
	    data_file << "\n" << "time: ";
	    data_file << std::setprecision(12) << tcurrent << " ";
	    data_file << "    dt: ";
	    data_file << std::setprecision(12) << dt << " ";
	    data_file << "\n";
	    for (int s(0); s < num_slip_; ++s) {
	      data_file << "\n" << "P" << s << ": ";
	      P = slip_systems_[s].projector_;
	      for (int i(0); i < num_dims_; ++i) {
		for (int j(0); j < num_dims_; ++j) {
		  data_file << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(P(i,j)) << " ";
		}
	      }
	    }
	    for (int s(0); s < num_slip_; ++s) {
	      data_file << "\n" << "slips: ";
	      data_file << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(slip_np1[s]) << " ";
	    }
	    data_file << "\n" << "F: ";
	    for (int i(0); i < num_dims_; ++i) {
	      for (int j(0); j < num_dims_; ++j) {
		data_file << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(F_np1(i,j)) << " ";
	      }
	    }
	    data_file << "\n" << "Fp: ";
	    for (int i(0); i < num_dims_; ++i) {
	      for (int j(0); j < num_dims_; ++j) {
		data_file << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(Fp_np1(i,j)) << " ";
	      }
	    }
	    data_file << "\n" << "Sigma: ";
	    for (int i(0); i < num_dims_; ++i) {
	      for (int j(0); j < num_dims_; ++j) {
		data_file << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(sigma_np1(i,j)) << " ";
	      }
	    }
	    data_file << "\n" << "Lp: ";
	    for (int i(0); i < num_dims_; ++i) {
	      for (int j(0); j < num_dims_; ++j) {
		data_file << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(Lp_np1(i,j)) << " ";
	      }
	    }
	    data_file << "\n";
	    data_file.close();
	  }
	} // end data file output
      } // end update_state_successful
    } // end loop over integration points
  } // end loop over elements
}

} // namespace LCM
