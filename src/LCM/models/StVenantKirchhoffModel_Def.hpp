//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
StVenantKirchhoffModel<EvalT, Traits>::StVenantKirchhoffModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("F", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("J", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // define the evaluated fields
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];
  this->eval_field_map_.insert(std::make_pair(cauchy, dl->qp_tensor));

  // define the state variables
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(true);
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
StVenantKirchhoffModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // extract dependent MDFields
  auto def_grad        = *dep_fields["F"];
  auto J               = *dep_fields["J"];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];
  // extract evaluated MDFields
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];
  auto        stress = *eval_fields[cauchy];
  ScalarT     lambda;
  ScalarT     mu;

  minitensor::Tensor<ScalarT> F(num_dims_), C(num_dims_), sigma(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor<ScalarT> S(num_dims_), E(num_dims_);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      lambda = (elastic_modulus(cell, pt) * poissons_ratio(cell, pt)) /
               (1. + poissons_ratio(cell, pt)) /
               (1 - 2 * poissons_ratio(cell, pt));
      mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      F.fill(def_grad, cell, pt, 0, 0);
      C     = F * transpose(F);
      E     = 0.5 * (C - I);
      S     = lambda * minitensor::trace(E) * I + 2.0 * mu * E;
      sigma = (1.0 / minitensor::det(F)) * F * S * minitensor::transpose(F);
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
