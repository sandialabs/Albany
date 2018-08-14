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
AnisotropicDamageModel<EvalT, Traits>::AnisotropicDamageModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      k_f1_(p->get<RealType>("Fiber 1 k", 1.0)),
      q_f1_(p->get<RealType>("Fiber 1 q", 1.0)),
      volume_fraction_f1_(p->get<RealType>("Fiber 1 volume fraction", 0.0)),
      max_damage_f1_(p->get<RealType>("Fiber 1 maximum damage", 1.0)),
      saturation_f1_(p->get<RealType>("Fiber 1 damage saturation", 0.0)),
      k_f2_(p->get<RealType>("Fiber 2 k", 1.0)),
      q_f2_(p->get<RealType>("Fiber 2 q", 1.0)),
      volume_fraction_f2_(p->get<RealType>("Fiber 2 volume fraction", 0.0)),
      max_damage_f2_(p->get<RealType>("Fiber 2 maximum damage", 1.0)),
      saturation_f2_(p->get<RealType>("Fiber 2 damage saturation", 0.0)),
      volume_fraction_m_(p->get<RealType>("Matrix volume fraction", 1.0)),
      max_damage_m_(p->get<RealType>("Matrix maximum damage", 1.0)),
      saturation_m_(p->get<RealType>("Matrix damage saturation", 0.0)),
      direction_f1_(
          p->get<Teuchos::Array<RealType>>("Fiber 1 Orientation Vector")
              .toVector()),
      direction_f2_(
          p->get<Teuchos::Array<RealType>>("Fiber 2 Orientation Vector")
              .toVector())
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("F", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("J", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // retrieve appropriate field name strings
  std::string cauchy_string        = (*field_name_map_)["Cauchy_Stress"];
  std::string matrix_energy_string = (*field_name_map_)["Matrix_Energy"];
  std::string f1_energy_string     = (*field_name_map_)["F1_Energy"];
  std::string f2_energy_string     = (*field_name_map_)["F2_Energy"];
  std::string matrix_damage_string = (*field_name_map_)["Matrix_Damage"];
  std::string f1_damage_string     = (*field_name_map_)["F1_Damage"];
  std::string f2_damage_string     = (*field_name_map_)["F2_Damage"];

  // optional material tangent computation
  std::string tangent_string = (*field_name_map_)["Material Tangent"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(
      std::make_pair(matrix_energy_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(f1_energy_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(f2_energy_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair(matrix_damage_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(f1_damage_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(f2_damage_string, dl->qp_scalar));

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
  // matrix energy
  this->num_state_variables_++;
  this->state_var_names_.push_back(matrix_energy_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // fiber 1 energy
  this->num_state_variables_++;
  this->state_var_names_.push_back(f1_energy_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // fiber 2 energy
  this->num_state_variables_++;
  this->state_var_names_.push_back(f2_energy_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // matrix damage
  this->num_state_variables_++;
  this->state_var_names_.push_back(matrix_damage_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // fiber 1 damage
  this->num_state_variables_++;
  this->state_var_names_.push_back(f1_damage_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // fiber 2 damage
  this->num_state_variables_++;
  this->state_var_names_.push_back(f2_damage_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
}
//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
AnisotropicDamageModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // bool print = false;
  // if (typeid(ScalarT) == typeid(RealType)) print = true;
  // cout.precision(15);

  // extract dependent MDFields
  auto def_grad        = *dep_fields["F"];
  auto J               = *dep_fields["J"];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];

  // retrieve appropriate field name strings
  std::string cauchy_string        = (*field_name_map_)["Cauchy_Stress"];
  std::string matrix_energy_string = (*field_name_map_)["Matrix_Energy"];
  std::string f1_energy_string     = (*field_name_map_)["F1_Energy"];
  std::string f2_energy_string     = (*field_name_map_)["F2_Energy"];
  std::string matrix_damage_string = (*field_name_map_)["Matrix_Damage"];
  std::string f1_damage_string     = (*field_name_map_)["F1_Damage"];
  std::string f2_damage_string     = (*field_name_map_)["F2_Damage"];
  std::string tangent_string       = (*field_name_map_)["Material Tangent"];

  // extract evaluated MDFields
  auto                  stress    = *eval_fields[cauchy_string];
  auto                  energy_m  = *eval_fields[matrix_energy_string];
  auto                  energy_f1 = *eval_fields[f1_energy_string];
  auto                  energy_f2 = *eval_fields[f2_energy_string];
  auto                  damage_m  = *eval_fields[matrix_damage_string];
  auto                  damage_f1 = *eval_fields[f1_damage_string];
  auto                  damage_f2 = *eval_fields[f2_damage_string];
  PHX::MDField<ScalarT> tangent;

  if (compute_tangent_) { tangent = *eval_fields[tangent_string]; }

  // previous state
  Albany::MDArray energy_m_old =
      (*workset.stateArrayPtr)[matrix_energy_string + "_old"];
  Albany::MDArray energy_f1_old =
      (*workset.stateArrayPtr)[f1_energy_string + "_old"];
  Albany::MDArray energy_f2_old =
      (*workset.stateArrayPtr)[f2_energy_string + "_old"];

  ScalarT mu, lame, mu_tilde, I1_m, I3_m, lnI3_m;
  ScalarT I4_f1, I4_f2;
  ScalarT alpha_f1, alpha_f2, alpha_m;
  ScalarT coefficient_f1, coefficient_f2;
  ScalarT damage_deriv_m, damage_deriv_f1, damage_deriv_f2;

  // Define some tensors for use
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor<ScalarT> F(num_dims_), C(num_dims_), invC(num_dims_);
  minitensor::Tensor<ScalarT> sigma_m(num_dims_), sigma_f1(num_dims_),
      sigma_f2(num_dims_);
  minitensor::Tensor<ScalarT> M1dyadM1(num_dims_), M2dyadM2(num_dims_);
  minitensor::Tensor<ScalarT> S0_m(num_dims_), S0_f1(num_dims_),
      S0_f2(num_dims_);

  // tangent is w.r.t. the right Cauchy-Green tensor
  minitensor::Tensor4<ScalarT> tangent_m(num_dims_);
  minitensor::Tensor4<ScalarT> tangent_f1(num_dims_), tangent_f2(num_dims_);
  // tangentA is w.r.t. the deformation gradient
  minitensor::Tensor4<ScalarT> tangentA_m(num_dims_);
  minitensor::Tensor4<ScalarT> tangentA_f1(num_dims_), tangentA_f2(num_dims_);

  minitensor::Vector<ScalarT> M1(num_dims_), M2(num_dims_);

  // volume_fraction_m_ = 1.0 - volume_fraction_f1_ - volume_fraction_f2_;

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      // for debugging
      // std::cout << "AnisotropicModel compute_tangent_=" << compute_tangent_
      // << std::endl;

      // local parameters
      mu = elastic_modulus(cell, pt) / (2.0 * (1.0 + poissons_ratio(cell, pt)));
      lame = elastic_modulus(cell, pt) * poissons_ratio(cell, pt) /
             (1.0 + poissons_ratio(cell, pt)) /
             (1.0 - 2.0 * poissons_ratio(cell, pt));

      F.fill(def_grad, cell, pt, 0, 0);
      // Right Cauchy-Green Tensor C = F^{T} * F

      C      = minitensor::transpose(F) * F;
      invC   = minitensor::inverse(C);
      I1_m   = minitensor::trace(C);
      I3_m   = minitensor::det(C);
      lnI3_m = std::log(I3_m);

      // energy for M
      energy_m(cell, pt) =
          volume_fraction_m_ * (0.125 * lame * lnI3_m * lnI3_m -
                                0.5 * mu * lnI3_m + 0.5 * mu * (I1_m - 3.0));

      // 2nd PK stress (undamaged) for M
      S0_m =
          volume_fraction_m_ * (0.5 * lame * lnI3_m * invC + mu * (I - invC));

      // undamaged Cauchy stress for M
      sigma_m =
          (1.0 / J(cell, pt)) *
          minitensor::dot(F, minitensor::dot(S0_m, minitensor::transpose(F)));

      // elasticity tensor (undamaged) for M
      mu_tilde = mu - 0.5 * lame * lnI3_m;

      // optional material tangent computation

      if (compute_tangent_) {
        tangent_m =
            volume_fraction_m_ * (lame * minitensor::tensor(invC, invC) +
                                  mu_tilde * (minitensor::tensor2(invC, invC) +
                                              minitensor::tensor3(invC, invC)));
      }  // compute_tangent_

      // damage term in M
      alpha_m = energy_m_old(cell, pt);
      if (energy_m(cell, pt) > alpha_m) alpha_m = energy_m(cell, pt);

      damage_m(cell, pt) =
          max_damage_m_ * (1 - std::exp(-alpha_m / saturation_m_));

      // derivative of damage w.r.t alpha
      damage_deriv_m =
          max_damage_m_ / saturation_m_ * std::exp(-alpha_m / saturation_m_);

      // optional material tangent computation
      if (compute_tangent_) {
        tangent_m = (1.0 - damage_m(cell, pt)) * tangent_m -
                    damage_deriv_m * minitensor::tensor(S0_m, S0_m);

        // convert tangent_m to tangentA
        // tangentA is w.r.t. the deformation gradient
        tangentA_m.fill(0.0);
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            for (int p(0); p < num_dims_; ++p) {
              for (int q(0); q < num_dims_; ++q) {
                for (int k(0); k < num_dims_; ++k) {
                  for (int n(0); n < num_dims_; ++n) {
                    tangentA_m(i, j, p, q) =
                        tangentA_m(i, j, p, q) +
                        F(i, k) * tangent_m(k, j, n, q) * F(p, n);
                  }
                }

                tangentA_m(i, j, p, q) =
                    tangentA_m(i, j, p, q) +
                    (1.0 - damage_m(cell, pt)) * S0_m(q, j) * I(i, p);
              }
            }
          }
        }

      }  // compute_tangent_

      //-----------compute quantities in Fibers------------

      // Fiber orientation vectors
      //
      // fiber 1
      for (int i = 0; i < num_dims_; ++i) { M1(i) = direction_f1_[i]; }
      M1 = M1 / norm(M1);

      // fiber 2
      for (int i = 0; i < num_dims_; ++i) { M2(i) = direction_f2_[i]; }
      M2 = M2 / norm(M2);

      // Anisotropic invariants I4 = M_{i} * C * M_{i}
      I4_f1    = minitensor::dot(M1, minitensor::dot(C, M1));
      I4_f2    = minitensor::dot(M2, minitensor::dot(C, M2));
      M1dyadM1 = minitensor::dyad(M1, M1);
      M2dyadM2 = minitensor::dyad(M2, M2);

      // compute energy for fibers
      energy_f1(cell, pt) =
          volume_fraction_f1_ *
          (k_f1_ * (std::exp(q_f1_ * (I4_f1 - 1) * (I4_f1 - 1)) - 1) / q_f1_);
      energy_f2(cell, pt) =
          volume_fraction_f2_ *
          (k_f2_ * (std::exp(q_f2_ * (I4_f2 - 1) * (I4_f2 - 1)) - 1) / q_f2_);

      // undamaged stress (2nd PK stress)
      S0_f1 = volume_fraction_f1_ *
              (4.0 * k_f1_ * (I4_f1 - 1.0) *
               std::exp(q_f1_ * (I4_f1 - 1) * (I4_f1 - 1))) *
              M1dyadM1;
      S0_f2 = volume_fraction_f2_ *
              (4.0 * k_f2_ * (I4_f2 - 1.0) *
               std::exp(q_f2_ * (I4_f2 - 1) * (I4_f2 - 1))) *
              M2dyadM2;

      // Fiber undamaged Cauchy stress
      sigma_f1 =
          (1.0 / J(cell, pt)) *
          minitensor::dot(F, minitensor::dot(S0_f1, minitensor::transpose(F)));
      sigma_f2 =
          (1.0 / J(cell, pt)) *
          minitensor::dot(F, minitensor::dot(S0_f2, minitensor::transpose(F)));

      // undamaged tangent for fibers
      coefficient_f1 = volume_fraction_f1_ * 8.0 * k_f1_ *
                       (1.0 + 2.0 * q_f1_ * (I4_f1 - 1.0) * (I4_f1 - 1.0)) *
                       exp(q_f1_ * (I4_f1 - 1.0) * (I4_f1 - 1.0));

      coefficient_f2 = volume_fraction_f2_ * 8.0 * k_f2_ *
                       (1.0 + 2.0 * q_f2_ * (I4_f2 - 1.0) * (I4_f2 - 1.0)) *
                       exp(q_f2_ * (I4_f2 - 1.0) * (I4_f2 - 1.0));

      // optional material tangent computation
      if (compute_tangent_) {
        tangent_f1 = coefficient_f1 * minitensor::tensor(M1dyadM1, M1dyadM1);
        tangent_f2 = coefficient_f2 * minitensor::tensor(M2dyadM2, M2dyadM2);
      }

      // maximum thermodynamic forces
      alpha_f1 = energy_f1_old(cell, pt);
      alpha_f2 = energy_f2_old(cell, pt);

      if (energy_f1(cell, pt) > alpha_f1) alpha_f1 = energy_f1(cell, pt);

      if (energy_f2(cell, pt) > alpha_f2) alpha_f2 = energy_f2(cell, pt);

      // damage term in fibers
      damage_f1(cell, pt) =
          max_damage_f1_ * (1 - std::exp(-alpha_f1 / saturation_f1_));
      damage_f2(cell, pt) =
          max_damage_f2_ * (1 - std::exp(-alpha_f2 / saturation_f2_));

      // derivative of damage w.r.t alpha
      damage_deriv_f1 = max_damage_f1_ / saturation_f1_ *
                        std::exp(-alpha_f1 / saturation_f1_);

      damage_deriv_f2 = max_damage_f2_ / saturation_f2_ *
                        std::exp(-alpha_f2 / saturation_f2_);

      // tangent for fibers including damage
      // optional material tangent computation
      if (compute_tangent_) {
        tangent_f1 = (1.0 - damage_f1(cell, pt)) * tangent_f1 -
                     damage_deriv_f1 * minitensor::tensor(S0_f1, S0_f1);
        tangent_f2 = (1.0 - damage_f2(cell, pt)) * tangent_f2 -
                     damage_deriv_f2 * minitensor::tensor(S0_f2, S0_f2);

        // convert tangent_m to tangentA
        // tangentA is w.r.t. the deformation gradient
        tangentA_f1.fill(0.0);
        tangentA_f2.fill(0.0);
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            for (int p(0); p < num_dims_; ++p) {
              for (int q(0); q < num_dims_; ++q) {
                for (int k(0); k < num_dims_; ++k) {
                  for (int n(0); n < num_dims_; ++n) {
                    tangentA_f1(i, j, p, q) =
                        tangentA_f1(i, j, p, q) +
                        F(i, k) * tangent_f1(k, j, n, q) * F(p, n);

                    tangentA_f2(i, j, p, q) =
                        tangentA_f2(i, j, p, q) +
                        F(i, k) * tangent_f2(k, j, n, q) * F(p, n);
                  }
                }

                tangentA_f1(i, j, p, q) =
                    tangentA_f1(i, j, p, q) +
                    (1.0 - damage_f1(cell, pt)) * S0_f1(q, j) * I(i, p);

                tangentA_f2(i, j, p, q) =
                    tangentA_f2(i, j, p, q) +
                    (1.0 - damage_f2(cell, pt)) * S0_f2(q, j) * I(i, p);
              }
            }
          }
        }

      }  // compute_tangent_

      // total Cauchy stress (M, Fibers)
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = (1.0 - damage_m(cell, pt)) * sigma_m(i, j) +
                                   (1. - damage_f1(cell, pt)) * sigma_f1(i, j) +
                                   (1. - damage_f2(cell, pt)) * sigma_f2(i, j);
        }
      }

      // total tangent (M, Fibers)
      if (compute_tangent_) {
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            for (int k(0); k < num_dims_; ++k) {
              for (int l(0); l < num_dims_; ++l) {
                // std::cout << "Tangent w.r.t. the deformation gradient"
                // << std::endl;
                tangent(cell, pt, i, j, k, l) = tangentA_m(i, j, k, l) +
                                                tangentA_f1(i, j, k, l) +
                                                tangentA_f2(i, j, k, l);

                // std::cout << "Tangent w.r.t. the right Cauchy-Green tensor"
                // << std::endl;
                // tangent(cell, pt, i, j, k, l) = tangent_m(i, j, k, l)
                // + tangent_f1(i, j, k, l) + tangent_f2(i, j, k, l);
              }  // l
            }    // k
          }      // j
        }        // i
      }          // compute_tangent_

    }  // pt
  }    // cell
}
//----------------------------------------------------------------------------
}  // namespace LCM
