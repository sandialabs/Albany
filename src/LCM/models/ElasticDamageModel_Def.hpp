//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>
#include <typeinfo>

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
ElasticDamageModel<EvalT, Traits>::ElasticDamageModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      max_damage_(p->get<RealType>("Maximum damage", 1.0)),
      saturation_(p->get<RealType>("Damage saturation", 0.0))
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Strain", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // retrieve appropriate field name strings
  std::string cauchy_string  = (*field_name_map_)["Cauchy_Stress"];
  std::string energy_string  = (*field_name_map_)["Matrix_Energy"];
  std::string damage_string  = (*field_name_map_)["Matrix_Damage"];
  std::string tangent_string = (*field_name_map_)["Material Tangent"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(energy_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(damage_string, dl->qp_scalar));

  if (compute_tangent_) {
    this->eval_field_map_.insert(
        std::make_pair(tangent_string, dl->qp_tensor4));
  }

  // define the state variables
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(true);
  //
  // energy
  this->num_state_variables_++;
  this->state_var_names_.push_back(energy_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // damage
  this->num_state_variables_++;
  this->state_var_names_.push_back(damage_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ElasticDamageModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // bool print = false;
  // if (typeid(ScalarT) == typeid(RealType)) print = true;
  // cout.precision(15);

  // extract dependent MDFields
  auto strain          = *dep_fields["Strain"];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];

  // retrieve appropriate field name strings
  std::string cauchy_string  = (*field_name_map_)["Cauchy_Stress"];
  std::string energy_string  = (*field_name_map_)["Matrix_Energy"];
  std::string damage_string  = (*field_name_map_)["Matrix_Damage"];
  std::string tangent_string = (*field_name_map_)["Material Tangent"];

  // extract evaluated MDFields
  auto                  stress = *eval_fields[cauchy_string];
  auto                  energy = *eval_fields[energy_string];
  auto                  damage = *eval_fields[damage_string];
  PHX::MDField<ScalarT> tangent;

  if (compute_tangent_) { tangent = *eval_fields[tangent_string]; }

  // previous state
  Albany::MDArray energy_old = (*workset.stateArrayPtr)[energy_string + "_old"];

  ScalarT mu, lame;
  ScalarT alpha, damage_deriv;

  // Define some tensors for use
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor<ScalarT> epsilon(num_dims_), sigma(num_dims_);

  minitensor::Tensor4<ScalarT> Ce(num_dims_);
  minitensor::Tensor4<ScalarT> id4(num_dims_);
  minitensor::Tensor4<ScalarT> id3(minitensor::identity_3<ScalarT>(num_dims_));

  id4 = 0.5 * (minitensor::identity_1<ScalarT>(num_dims_) +
               minitensor::identity_2<ScalarT>(num_dims_));

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      // local parameters
      mu = elastic_modulus(cell, pt) / (2.0 * (1.0 + poissons_ratio(cell, pt)));
      lame = elastic_modulus(cell, pt) * poissons_ratio(cell, pt) /
             (1.0 + poissons_ratio(cell, pt)) /
             (1.0 - 2.0 * poissons_ratio(cell, pt));

      // small strain tensor
      epsilon.fill(strain, cell, pt, 0, 0);

      // undamaged elasticity tensor
      Ce = lame * id3 + 2.0 * mu * id4;

      // undamaged energy
      energy(cell, pt) =
          0.5 * minitensor::dotdot(minitensor::dotdot(epsilon, Ce), epsilon);

      // undamaged Cauchy stress
      sigma = minitensor::dotdot(Ce, epsilon);

      // maximum thermodynamic force
      alpha = energy_old(cell, pt);
      if (energy(cell, pt) > alpha) alpha = energy(cell, pt);

      // damage term
      damage(cell, pt) = max_damage_ * (1 - std::exp(-alpha / saturation_));

      // derivative of damage w.r.t alpha
      damage_deriv = max_damage_ / saturation_ * std::exp(-alpha / saturation_);

      // tangent for matrix considering damage
      if (compute_tangent_) {
        Ce = (1.0 - damage(cell, pt)) * Ce -
             damage_deriv * minitensor::tensor(sigma, sigma);
      }

      // total Cauchy stress
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = (1.0 - damage(cell, pt)) * sigma(i, j);
        }
      }

      // total tangent
      if (compute_tangent_) {
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            for (int k(0); k < num_dims_; ++k) {
              for (int l(0); l < num_dims_; ++l) {
                tangent(cell, pt, i, j, k, l) = Ce(i, j, k, l);
              }
            }
          }
        }
      }  // compute_tangent_

    }  // pt
  }    // cell
}
//----------------------------------------------------------------------------
}  // namespace LCM
