//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

//----------------------------------------------------------------------------
template<typename EvalT, typename Traits>
NeohookeanModel<EvalT, Traits>::
NeohookeanModel(Teuchos::ParameterList* p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
{
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];

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
  this->state_var_output_flags_.push_back(p->get<bool>("Output Cauchy Stress", false));
}
//----------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void NeohookeanModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad = *dep_fields[F_string];
  PHX::MDField<ScalarT> J = *dep_fields[J_string];
  PHX::MDField<ScalarT> poissons_ratio = *dep_fields["Poissons Ratio"];
  PHX::MDField<ScalarT> elastic_modulus = *dep_fields["Elastic Modulus"];
  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy];
  PHX::MDField<ScalarT> energy = *eval_fields["Energy"];
  PHX::MDField<ScalarT> tangent = *eval_fields["Material Tangent"];
  ScalarT kappa;
  ScalarT mu, mubar;
  ScalarT Jm53, Jm23;
  ScalarT smag;

  Intrepid::Tensor<ScalarT> F(num_dims_), b(num_dims_), sigma(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
  Intrepid::Tensor<ScalarT> s(num_dims_), n(num_dims_);

  Intrepid::Tensor4<ScalarT> dsigmadb;
  Intrepid::Tensor4<ScalarT> I1(Intrepid::identity_1<ScalarT>(num_dims_));
  Intrepid::Tensor4<ScalarT> I3(Intrepid::identity_3<ScalarT>(num_dims_));

  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      kappa =
          elastic_modulus(cell, pt)
              / (3. * (1. - 2. * poissons_ratio(cell, pt)));
      mu =
          elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      Jm53 = std::pow(J(cell, pt), -5. / 3.);
      Jm23 = Jm53 * J(cell, pt);

      F.fill(&def_grad(cell, pt, 0, 0));
      b = F * transpose(F);
      mubar = (1.0 / 3.0) * mu * Jm23 * Intrepid::trace(b);

      sigma = 0.5 * kappa * (J(cell, pt) - 1. / J(cell, pt)) * I
          + mu * Jm53 * Intrepid::dev(b);

      for (std::size_t i = 0; i < num_dims_; ++i) {
        for (std::size_t j = 0; j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }

      if (compute_energy_) { // compute energy
        energy(cell, pt) =
            0.5 * kappa
                * (0.5 * (J(cell, pt) * J(cell, pt) - 1.0)
                    - std::log(J(cell, pt)))
                + 0.5 * mu * (Jm23 * Intrepid::trace(b) - 3.0);
      }

      if (compute_tangent_) { // compute tangent

        s = Intrepid::dev(sigma);
        smag = Intrepid::norm(s);
        n = s / smag;

        dsigmadb =
            kappa * J(cell, pt) * J(cell, pt) * I3
                - kappa * (J(cell, pt) * J(cell, pt) - 1.0) * I1
                + 2.0 * mubar * (I1 - (1.0 / 3.0) * I3)
                - 2.0 / 3.0 * smag
                    * (Intrepid::tensor(n, I) + Intrepid::tensor(I, n));

        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            for (std::size_t k = 0; k < num_dims_; ++k) {
              for (std::size_t l = 0; l < num_dims_; ++l) {
                tangent(cell, pt, i, j, k, l) = dsigmadb(i, j, k, l);
              }
            }
          }
        }
      }
    }
  }

  if (have_temperature_) {
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        F.fill(&def_grad(cell,pt,0,0));
        ScalarT J = Intrepid::det(F);
        sigma.fill(&stress(cell,pt,0,0));
        sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J*J))
          * (temperature_(cell,pt) - ref_temperature_) * I;

        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            stress(cell, pt, i, j) = sigma(i, j);
          }
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}

