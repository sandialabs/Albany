//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_Utils.hpp"
#include "MiniTensor.h"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

//#define PRINT_DEBUG

namespace LCM {

//------------------------------------------------------------------------------
// See Ortiz and Pandolfi, IJNME (1999)
// Finite-deformation irreversible cohesive elements for 3-D crack
// propagation analysis
//------------------------------------------------------------------------------

template <typename EvalT, typename Traits>
OrtizPandolfiModel<EvalT, Traits>::OrtizPandolfiModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
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
template <typename EvalT, typename Traits>
void
OrtizPandolfiModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // extract dependent MDFields
  auto mdf_jump  = *dep_fields["Vector Jump"];
  auto mdf_basis = *dep_fields["Current Basis"];

  // extract evaluated MDFields
  auto mdf_traction        = *eval_fields["Cohesive_Traction"];
  auto mdf_traction_normal = *eval_fields["Normal_Traction"];
  auto mdf_traction_shear  = *eval_fields["Shear_Traction"];
  auto mdf_jump_normal     = *eval_fields["Normal_Jump"];
  auto mdf_jump_shear      = *eval_fields["Shear_Jump"];
  auto mdf_jump_max        = *eval_fields["Max_Jump"];

  // get state variable
  Albany::MDArray jump_max_old = (*workset.stateArrayPtr)["Max_Jump_old"];

  bool print_debug = false;
#if defined(PRINT_DEBUG)
  if (typeid(ScalarT) == typeid(RealType)) { print_debug = true; }
  std::cout.precision(15);
#endif

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      // current basis vector
      minitensor::Vector<ScalarT> g_0(
          minitensor::Source::ARRAY, 3, mdf_basis, cell, pt, 0, 0);

      minitensor::Vector<ScalarT> g_1(
          minitensor::Source::ARRAY, 3, mdf_basis, cell, pt, 1, 0);

      minitensor::Vector<ScalarT> n(
          minitensor::Source::ARRAY, 3, mdf_basis, cell, pt, 2, 0);

      // current jump vector - move PHX::MDField into minitensor::Vector
      minitensor::Vector<ScalarT> jump_pt(
          minitensor::Source::ARRAY, 3, mdf_jump, cell, pt, 0);

      // construct Identity tensor (2nd order) and tensor product of normal
      minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(3));
      minitensor::Tensor<ScalarT> Fn(minitensor::bun(n, n));

      // define components of the jump
      // jump_n is the normal component
      // jump_s is the shear component
      // jump_m is the maximum effective jump from prior converged iteration
      // vec_jump_s is the shear vector
      ScalarT                     jump_m     = jump_max_old(cell, pt);
      ScalarT                     jump_n     = minitensor::dot(jump_pt, n);
      minitensor::Vector<ScalarT> vec_jump_s = minitensor::dot(I - Fn, jump_pt);
      // Be careful regarding Sacado and sqrt()
      ScalarT const jump_s2 = minitensor::dot(vec_jump_s, vec_jump_s);

      ScalarT jump_s = 0.0;
      if (jump_s2 > 0.0) { jump_s = std::sqrt(jump_s2); }

      // define the effective jump
      // for interpenetration, only employ shear component

      // Default no effective jump.
      ScalarT jump_eff = 0.0;

      if (jump_n >= 0.0) {
        // Be careful regarding Sacado and sqrt()
        ScalarT const jump_eff2 =
            beta * beta * jump_s * jump_s + jump_n * jump_n;

        if (jump_eff2 > 0.0) { jump_eff = std::sqrt(jump_eff2); }
      } else {
        jump_eff = beta * jump_s;
      }

      // Debugging - print kinematics
      if (print_debug) {
        std::cout << "--- KINEMATICS CELL: " << cell << ", IP: " << pt << '\n';
        std::cout << "d    : " << jump_pt << '\n';
        std::cout << "d_n  : " << jump_n << '\n';
        std::cout << "d_s  : " << jump_s << '\n';
        std::cout << "d_eff: " << jump_eff << '\n';
      }

      // define the constitutive response through an effective traction

      // Default completely unloaded
      ScalarT t_eff = 0.0;

      if (jump_eff < delta_c) {
        if (jump_eff >= jump_m) {
          // linear unloading toward delta_c
          t_eff = sigma_c * (1.0 - jump_eff / delta_c);
        } else {
          // linear unloading toward origin
          t_eff = sigma_c * (jump_eff / jump_m - jump_eff / delta_c);
        }
      }

      // calculate the global traction
      // penalize interpenetration through stiff_c

      // Normal traction, default to zero.
      minitensor::Vector<ScalarT> traction_normal(3, minitensor::Filler::ZEROS);

      if (jump_n >= 0.0) {
        ALBANY_EXPECT(jump_eff >= 0.0);

        if (jump_n > 0.0) {
          ALBANY_EXPECT(jump_eff > 0.0);
          traction_normal = t_eff / jump_eff * jump_n * n;
        } else {
          // FIXME: Assume that if there is no jump whatever (initial state)
          // the initial traction will all be in the normal direction.
          // Could pass on traction information from insertion criterion
          // to determine a better direction and avoid a big residual at
          // insertion but not sure it will have a big effect in the end
          // on the solver.
          traction_normal = t_eff * n;
        }

      } else {
        // Interpenetration
        traction_normal = stiff_c * jump_n * n;
      }

      // Shear traction, default to zero.
      minitensor::Vector<ScalarT> traction_shear(3, minitensor::Filler::ZEROS);

      if (jump_eff > 0.0) {
        traction_shear = t_eff / jump_eff * beta * beta * vec_jump_s;
      }

      minitensor::Vector<ScalarT> traction_vector =
          traction_normal + traction_shear;

      // Debugging - debug_print tractions
      if (print_debug) {
        std::cout << "--- TRACTION CELL: " << cell << ", IP: " << pt << '\n';
        std::cout << "t    : " << traction_vector << '\n';
        std::cout << "t_eff: " << t_eff << '\n';
      }

      // update global traction
      mdf_traction(cell, pt, 0) = traction_vector(0);
      mdf_traction(cell, pt, 1) = traction_vector(1);
      mdf_traction(cell, pt, 2) = traction_vector(2);

      // update state variables

      // Calculate normal and shear components of global traction
      ScalarT traction_n = minitensor::dot(traction_vector, n);
      minitensor::Vector<ScalarT> vec_traction_s =
          minitensor::dot(I - Fn, jump_pt);
      // Be careful regarding Sacado and sqrt()
      ScalarT traction_s2 = minitensor::dot(vec_traction_s, vec_traction_s);
      ScalarT traction_s  = 0.0;
      if (traction_s2 > 0.0) { traction_s = std::sqrt(traction_s2); }
      mdf_traction_normal(cell, pt) = traction_n;
      mdf_traction_shear(cell, pt)  = traction_s;

      // Populate mdf_jump_normal and mdf_jump_shear
      mdf_jump_normal(cell, pt) = jump_n;
      mdf_jump_shear(cell, pt)  = jump_s;

      // only true state variable is mdf_jump_max
      if (jump_eff > jump_m) { mdf_jump_max(cell, pt) = jump_eff; }
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
