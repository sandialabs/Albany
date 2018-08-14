//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Author: Mario J. Juha (juham@rpi.edu)

#include <cmath>
#include "MiniTensor.h"
#include "MiniTensor_Definitions.h"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
ViscoElasticModel<EvalT, Traits>::ViscoElasticModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
{
  // Read elastic coefficients
  Teuchos::ParameterList e_list = p->sublist("Relaxation time");
  //
  tau1_ = e_list.get<RealType>("tau1");
  tau2_ = e_list.get<RealType>("tau2");
  tau3_ = e_list.get<RealType>("tau3");

  // Read stiffness ratio
  e_list     = p->sublist("Stiffness ratio");
  gamma1_    = e_list.get<RealType>("gamma1");
  gamma2_    = e_list.get<RealType>("gamma2");
  gamma3_    = e_list.get<RealType>("gamma3");
  gamma_inf_ = e_list.get<RealType>("gamma_inf");

  // Read shear modulus
  e_list = p->sublist("Shear modulus");
  mu_    = e_list.get<RealType>("mu");

  // Read gas constant
  e_list = p->sublist("Gas Constant");
  R_     = e_list.get<RealType>("R");

  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy   = (*field_name_map_)["Cauchy_Stress"];
  // this variable will store the instantaneous stress
  std::string S0_string = (*field_name_map_)["Instantaneous Stress"];
  // This variable is used to store state variable H
  std::string H1_string = (*field_name_map_)["H_1"];
  std::string H2_string = (*field_name_map_)["H_2"];
  std::string H3_string = (*field_name_map_)["H_3"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(S0_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(H1_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(H2_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(H3_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair("Time", dl->workset_scalar));

  // define the state variables
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));

  // Instantaneous stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(S0_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Instantaneous Stress", false));

  // H1
  this->num_state_variables_++;
  this->state_var_names_.push_back(H1_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output H_1", false));

  // H2
  this->num_state_variables_++;
  this->state_var_names_.push_back(H2_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output H_2", false));

  // H3
  this->num_state_variables_++;
  this->state_var_names_.push_back(H3_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output H_3", false));
  //
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ViscoElasticModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // Get names
  std::string F_string  = (*field_name_map_)["F"];
  std::string J_string  = (*field_name_map_)["J"];
  std::string cauchy    = (*field_name_map_)["Cauchy_Stress"];
  std::string S0_string = (*field_name_map_)["Instantaneous Stress"];
  std::string H1_string = (*field_name_map_)["H_1"];
  std::string H2_string = (*field_name_map_)["H_2"];
  std::string H3_string = (*field_name_map_)["H_3"];

  // std::cout << workset.current_time << std::endl;
  // std::cout << workset.previous_time << std::endl;
  // ScalarT dt = 1.0e-7;

  //
  // extract dependent MDFields
  //
  auto def_grad   = *dep_fields[F_string];
  auto J          = *dep_fields[J_string];
  auto delta_time = *dep_fields["Delta Time"];

  //
  // extract evaluated MDFields
  //
  auto stress = *eval_fields[cauchy];
  // S_0
  auto stress_0 = *eval_fields[S0_string];
  // H_alpha
  auto H1 = *eval_fields[H1_string];
  auto H2 = *eval_fields[H2_string];
  auto H3 = *eval_fields[H3_string];
  // time
  auto time = *eval_fields["Time"];

  //
  // Extract previous values (state variables)
  //
  Albany::MDArray J_old        = (*workset.stateArrayPtr)[J_string + "_old"];
  Albany::MDArray stress_0_old = (*workset.stateArrayPtr)[S0_string + "_old"];
  Albany::MDArray H1_old       = (*workset.stateArrayPtr)[H1_string + "_old"];
  Albany::MDArray H2_old       = (*workset.stateArrayPtr)[H2_string + "_old"];
  Albany::MDArray H3_old       = (*workset.stateArrayPtr)[H3_string + "_old"];

  // get dt
  RealType dt = Sacado::ScalarValue<ScalarT>::eval(delta_time(0));
  // get current time
  RealType tcurrent = Sacado::ScalarValue<ScalarT>::eval(time(0));

  //    std::cout << "dt = " << dt << std::endl;
  //    std::cout << "tcurrent = " << tcurrent << std::endl;

  // deformation gradient
  minitensor::Tensor<ScalarT> F(num_dims_);

  // deformation gradient old
  minitensor::Tensor<ScalarT> F_old(num_dims_);

  // Inverse deformation gradient
  minitensor::Tensor<ScalarT> Finv(num_dims_);

  // Right Cauchy-Green deformation tensor (do not confuse with C_). C = F^{T}*F
  minitensor::Tensor<ScalarT> C(num_dims_);

  // Inverse of Cauchy-Green deformation tensor.
  minitensor::Tensor<ScalarT> Cinv(num_dims_);

  // Right Cauchy-Green deformation tensor times J^{-2/3}. C23 = J^{-2/3}*C
  minitensor::Tensor<ScalarT> C23(num_dims_);

  // Inverse of Cauchy-Green deformation tensor.
  minitensor::Tensor<ScalarT> C23inv(num_dims_);

  // First Piola-Kirchhoff stress
  minitensor::Tensor<ScalarT> PK(num_dims_);

  // sigma (Cauchy stress)
  minitensor::Tensor<ScalarT> sigma(num_dims_);

  // S0 (Instantaneous stress)
  minitensor::Tensor<ScalarT> S0(num_dims_);

  // S0_old (Instantaneous stress old)
  minitensor::Tensor<ScalarT> S0_old(num_dims_);

  // State variables
  minitensor::Tensor<ScalarT> h1(num_dims_);
  minitensor::Tensor<ScalarT> h2(num_dims_);
  minitensor::Tensor<ScalarT> h3(num_dims_);

  // State variables old
  minitensor::Tensor<ScalarT> h1_old(num_dims_);
  minitensor::Tensor<ScalarT> h2_old(num_dims_);
  minitensor::Tensor<ScalarT> h3_old(num_dims_);

  // state variable alpha
  minitensor::Tensor<ScalarT> h1_alpha(num_dims_);
  minitensor::Tensor<ScalarT> h2_alpha(num_dims_);
  minitensor::Tensor<ScalarT> h3_alpha(num_dims_);

  //
  //    // Temporal variables
  //    minitensor::Tensor<ScalarT> tmp1(num_dims_);

  minitensor::Tensor<ScalarT> Dev_Stress(num_dims_);

  // Jacobian
  ScalarT Jac;

  // Jacobian old
  ScalarT Jac_old;

  // Jacobian^{-2/3}
  ScalarT Jac23_inv;

  // Jacobian^{2/3}
  ScalarT Jac23;

  // Jacobian^{2/3}_old
  ScalarT Jac23_old;

  // p_star = \rho * R * T
  ScalarT p_star;

  // p_0 = \rho_0 * R * T_0
  ScalarT p_0;

  // compute initial pressure
  p_0 = density_ * R_ * ref_temperature_;

  // pressure = p_start - p_0
  ScalarT pressure;

  // Identity tensor
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      // get Jacobian
      Jac = J(cell, pt);
      // get Jacobian old
      Jac_old = J_old(cell, pt);
      // get Jac23 at Gauss point
      Jac23_inv = std::pow(Jac, -2.0 / 3.0);
      //
      Jac23 = std::pow(Jac, 2.0 / 3.0);
      //
      Jac23_old = std::pow(Jac_old, 2.0 / 3.0);
      // Fill deformation gradient
      F.fill(def_grad, cell, pt, 0, 0);
      //             Fill old values
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          S0_old(i, j) = stress_0_old(cell, pt, i, j);
          h1_old(i, j) = H1_old(cell, pt, i, j);
          h2_old(i, j) = H2_old(cell, pt, i, j);
          h3_old(i, j) = H3_old(cell, pt, i, j);
        }
      }
      // compute right Cauchy-Green deformation tensor ==> C = F^{T}*F
      C = transpose(F) * F;
      // compute inverse of C
      Cinv = minitensor::inverse(C);
      // compute modified right Cauchy-Green deformation tensor ==> C =
      // J^{-2/3}*F^{T}*F
      C23 = Jac23_inv * C;
      // compute inverse of C
      C23inv = minitensor::inverse(C23);
      // Inverse deformation gradient
      Finv = minitensor::inverse(F);

      // compute instantaneous stress
      S0 =
          Jac23_inv * mu_ * (I - (1.0 / 3.0) * minitensor::trace(C23) * C23inv);

      // Compute state variables h_alpha
      h1 = exp(-dt / tau1_) * h1_old +
           exp(-0.5 * dt / tau1_) * (Jac23 * S0 - Jac23_old * S0_old);
      h2 = exp(-dt / tau2_) * h2_old +
           exp(-0.5 * dt / tau2_) * (Jac23 * S0 - Jac23_old * S0_old);
      h3 = exp(-dt / tau3_) * h1_old +
           exp(-0.5 * dt / tau3_) * (Jac23 * S0 - Jac23_old * S0_old);

      // compute state variables h_bar
      ScalarT sum1(0.0);
      ScalarT sum2(0.0);
      ScalarT sum3(0.0);
      for (int i(0); i < num_dims_; i++) {
        for (int j(0); j < num_dims_; j++) {
          sum1 = sum1 + h1(i, j) * C(i, j);
          sum2 = sum2 + h2(i, j) * C(i, j);
          sum3 = sum3 + h3(i, j) * C(i, j);
        }
      }
      h1_alpha = h1 - (1.0 / 3.0) * (sum1)*Cinv;
      h2_alpha = h2 - (1.0 / 3.0) * (sum2)*Cinv;
      h3_alpha = h3 - (1.0 / 3.0) * (sum3)*Cinv;

      // Deviator stress
      Dev_Stress = gamma_inf_ * S0 +
                   Jac23_inv * (gamma1_ * h1_alpha + gamma2_ * h2_alpha +
                                gamma3_ * h3_alpha);

      // compute p_0 using gas law
      p_star = density_ * (1.0 / Jac) * R_ * temperature_(cell, pt);

      // compute pressure
      pressure = p_star - p_0;

      // compute first Piola-Kirchhoff stress tensor
      PK = F * Dev_Stress - Jac * pressure * transpose(Finv);

      // transform it to Cauchy stress (true stress)
      sigma = (1.0 / Jac) * PK * transpose(F);

      // fill Cauchy stress and instantaneous stress
      for (int i = 0; i < num_dims_; i++) {
        for (int j = 0; j < num_dims_; j++) {
          stress(cell, pt, i, j)   = sigma(i, j);
          stress_0(cell, pt, i, j) = S0(i, j);
          H1(cell, pt, i, j)       = h1(i, j);
          H2(cell, pt, i, j)       = h2(i, j);
          H3(cell, pt, i, j)       = h3(i, j);
        }
      }

    }  // end pt
  }    // end cell
}
//----------------------------------------------------------------------------

}  // namespace LCM
