//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//
//
//
template <typename EvalT, typename Traits>
NeohookeanModel<EvalT, Traits>::NeohookeanModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
{
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy   = (*field_name_map_)["Cauchy_Stress"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair("Energy", dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair("Material Tangent", dl->qp_tensor4));

  // define the state variables
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));
}

//
//
//
template <typename EvalT, typename Traits>
void
NeohookeanModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy   = (*field_name_map_)["Cauchy_Stress"];

  // extract dependent MDFields
  auto def_grad        = *dep_fields[F_string];
  auto jac_det         = *dep_fields[J_string];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];

  // extract evaluated MDFields
  auto stress  = *eval_fields[cauchy];
  auto energy  = *eval_fields["Energy"];
  auto tangent = *eval_fields["Material Tangent"];

  ScalarT kappa;
  ScalarT mu, mubar;
  ScalarT Jm13, Jm53, Jm23;
  ScalarT smag;

  minitensor::Tensor<ScalarT> F(num_dims_), b(num_dims_), sigma(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor<ScalarT> s(num_dims_), n(num_dims_);

  minitensor::Tensor4<ScalarT> dsigmadb;
  minitensor::Tensor4<ScalarT> I1(minitensor::identity_1<ScalarT>(num_dims_));
  minitensor::Tensor4<ScalarT> I3(minitensor::identity_3<ScalarT>(num_dims_));

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      auto const& E = elastic_modulus(cell, pt);

      auto const& nu = poissons_ratio(cell, pt);

      auto const& J = jac_det(cell, pt);

      kappa = E / (3.0 * (1.0 - 2.0 * nu));

      mu = E / (2.0 * (1.0 + nu));

      Jm13 = 1.0 / std::cbrt(J);

      Jm23 = Jm13 * Jm13;

      Jm53 = Jm23 * Jm23 * Jm13;

      F.fill(def_grad, cell, pt, 0, 0);

      // Mechanical deformation gradient
      auto Fm = minitensor::Tensor<ScalarT>(F);
      if (have_temperature_) {
        // Compute the mechanical deformation gradient Fm based on the
        // multiplicative decomposition of the deformation gradient
        //
        //            F = Fm.Ft => Fm = F.inv(Ft)
        //
        // where Ft is the thermal part of F, given as
        //
        //     Ft = Le * I = exp(alpha * dtemp) * I
        //
        // Le = exp(alpha*dtemp) is the thermal stretch and alpha the
        // coefficient of thermal expansion.
        ScalarT dtemp           = temperature_(cell, pt) - ref_temperature_;
        ScalarT thermal_stretch = std::exp(expansion_coeff_ * dtemp);
        Fm /= thermal_stretch;
      }
      b     = Fm * minitensor::transpose(Fm);
      mubar = (1.0 / 3.0) * mu * Jm23 * minitensor::trace(b);
      sigma = 0.5 * kappa * (J - 1.0 / J) * I + mu * Jm53 * minitensor::dev(b);

      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }

      if (compute_energy_ == true) {
        energy(cell, pt) = 0.5 * kappa * (0.5 * (J * J - 1.0) - std::log(J)) +
                           0.5 * mu * (Jm23 * minitensor::trace(b) - 3.0);
      }

      if (compute_tangent_ == true) {
        s    = minitensor::dev(sigma);
        smag = minitensor::norm(s);
        n    = s / smag;

        dsigmadb = kappa * J * J * I3 - kappa * (J * J - 1.0) * I1 +
                   2.0 * mubar * (I1 - (1.0 / 3.0) * I3) -
                   2.0 / 3.0 * smag *
                       (minitensor::tensor(n, I) + minitensor::tensor(I, n));

        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            for (int k = 0; k < num_dims_; ++k) {
              for (int l = 0; l < num_dims_; ++l) {
                tangent(cell, pt, i, j, k, l) = dsigmadb(i, j, k, l);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace LCM
