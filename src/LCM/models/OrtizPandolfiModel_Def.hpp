//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>

#define PRINT_DEBUG

namespace LCM
{

//------------------------------------------------------------------------------
// See Ortiz and Pandolfi, IJNME (1999)
// Finite-deformation irreversible cohesive elements for 3-D crack
// propagation analysis
//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
OrtizPandolfiModel<EvalT, Traits>::
OrtizPandolfiModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    delta_c(p->get<RealType>("delta_c", 1.0)),
    sigma_c(p->get<RealType>("sigma_c", 1.0)),
    beta(p->get<RealType>("beta", 1.0)),
    stiff_c(p->get<RealType>("stiff_c", 1.0))
{

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Vector Jump", dl->qp_vector));
  this->dep_field_map_.insert(std::make_pair("Current Basis", dl->qp_tensor));

  // define the evaluated fields
  this->eval_field_map_.insert(
      std::make_pair("Cohesive_Traction", dl->qp_vector));
  this->eval_field_map_.insert(
      std::make_pair("Normal_Traction", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Shear_Traction", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Normal_Jump", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Shear_Jump", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Max_Jump", dl->qp_scalar));

  // define the state variables
  //
  // cohesive traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Cohesive_Traction");
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cohesive Traction", false));
  //
  // normal traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Normal_Traction");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Normal Traction", false));
  //
  // shear traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Shear_Traction");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Shear Traction", false));
  //
  // normal jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Normal_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Normal Jump", false));
  //
  // shear jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Shear_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Shear Jump", false));
  //
  // max jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Max_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Max Jump", false));
  //
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void OrtizPandolfiModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{

  // extract dependent MDFields
  PHX::MDField<ScalarT> jump = *dep_fields["Vector Jump"];
  PHX::MDField<ScalarT> basis = *dep_fields["Current Basis"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> traction = *eval_fields["Cohesive_Traction"];
  PHX::MDField<ScalarT> traction_normal = *eval_fields["Normal_Traction"];
  PHX::MDField<ScalarT> traction_shear = *eval_fields["Shear_Traction"];
  PHX::MDField<ScalarT> jump_normal = *eval_fields["Normal_Jump"];
  PHX::MDField<ScalarT> jump_shear = *eval_fields["Shear_Jump"];
  PHX::MDField<ScalarT> jump_max = *eval_fields["Max_Jump"];

  // get state variable
  Albany::MDArray jump_max_old = (*workset.stateArrayPtr)["Max_Jump_old"];

  bool print_debug = false;
#if defined(PRINT_DEBUG)
  if (typeid(ScalarT) == typeid(RealType)) {
    print_debug = true;
  }
  std::cout.precision(15);
#endif

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {

      //current basis vector
      Intrepid::Vector<ScalarT> g_0(3, basis, cell, pt, 0, 0);
      Intrepid::Vector<ScalarT> g_1(3, basis, cell, pt, 1, 0);
      Intrepid::Vector<ScalarT> n(3, basis, cell, pt, 2, 0);

      //current jump vector - move PHX::MDField into Intrepid::Vector
      Intrepid::Vector<ScalarT> jump_pt(3, jump, cell, pt, 0);

      //construct Identity tensor (2nd order) and tensor product of normal
      Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(3));
      Intrepid::Tensor<ScalarT> Fn(Intrepid::bun(n, n));

      // define components of the jump
      // jump_n is the normal component
      // jump_s is the shear component
      // jump_m is the maximum effective jump from prior converged iteration
      // vec_jump_s is the shear vector
      ScalarT jump_m = jump_max_old(cell, pt);
      ScalarT jump_n = Intrepid::dot(jump_pt, n);
      Intrepid::Vector<ScalarT> vec_jump_s = Intrepid::dot(I - Fn, jump_pt);
      ScalarT jump_s = sqrt(Intrepid::dot(vec_jump_s, vec_jump_s));

      // define the effective jump
      // for intepenetration, only employ shear component

      ScalarT jump_eff;
      if (jump_n >= 0.0) {
        jump_eff = sqrt(beta * beta * jump_s * jump_s + jump_n * jump_n);
      }
      else {
        jump_eff = beta * jump_s;
      }

      // Debugging - print kinematics
      if (print_debug) {
        std::cout << "jump for cell " << cell << " integration point " << pt
            << '\n';
        std::cout << jump_pt << '\n';
        std::cout << "normal jump for cell " << cell << " integration point "
            << pt << '\n';
        std::cout << jump_n << '\n';
        std::cout << "shear jump for cell " << cell << " integration point "
            << pt << '\n';
        std::cout << jump_s << '\n';
        std::cout << "effective jump for cell " << cell << " integration point "
            << pt << '\n';
        std::cout << jump_eff << '\n';
      }

      // define the constitutive response through an effective traction

      ScalarT t_eff;
      if (jump_eff < jump_m && jump_eff < delta_c) {
        // linear unloading toward origin
        t_eff = sigma_c / jump_m * (1.0 - jump_m / delta_c) * jump_eff;
      }
      else if (jump_eff >= jump_m && jump_eff <= delta_c) {
        // linear unloading toward delta_c
        t_eff = sigma_c * (1.0 - jump_eff / delta_c);
      }
      else {
        // completely unloaded
        t_eff = 0.0;
      }

      // calculate the global traction
      // penalize interpentration through stiff_c
      Intrepid::Vector<ScalarT> t_vec(3);
      if (jump_n == 0.0 & jump_eff == 0.0) {
        // no interpenetration, no effective jump
        t_vec = 0.0 * n;
      }
      else if (jump_n < 0.0 && jump_eff == 0.0) {
        // interpenetration, no effective jump
        t_vec = stiff_c * jump_n * n;
      }
      else if (jump_n < 0.0 && jump_eff > 0.0) {
        //  interpenetration, effective jump
        t_vec = t_eff / jump_eff * beta * beta * vec_jump_s
            + stiff_c * jump_n * n;
      }
      else {
        t_vec = t_eff / jump_eff * (beta * beta * vec_jump_s + jump_n * n);
      }

      // Debugging - debug_print tractions
      if (print_debug) {
        std::cout << "traction for cell " << cell << " integration point " << pt
            << '\n';
        std::cout << t_vec << '\n';
        std::cout << "effective traction for cell " << cell
            << " integration point " << pt << '\n';
        std::cout << t_eff << '\n';
      }

      // update global traction
      traction(cell, pt, 0) = t_vec(0);
      traction(cell, pt, 1) = t_vec(1);
      traction(cell, pt, 2) = t_vec(2);

      // update state variables 
      if (jump_n < 0.0) {
        traction_normal(cell, pt) = stiff_c * jump_n;
      }
      else {
        traction_normal(cell, pt) = t_eff * jump_n / jump_eff;
      }

      traction_shear(cell, pt) = t_eff * jump_s / jump_eff * beta * beta;
      jump_normal(cell, pt) = jump_n;
      jump_shear(cell, pt) = jump_s;

      // only true state variable is jump_max
      if (jump_eff > jump_m) {
        jump_max(cell, pt) = jump_eff;
      }

    }
  }
}
//------------------------------------------------------------------------------
}

