//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <typeinfo>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"
namespace LCM {

//
//
//
template <typename EvalT, typename Traits>
LinearElasticVolDevModel<EvalT, Traits>::LinearElasticVolDevModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
{
  // Baseline constants
  bulk_modulus_  = p->get<RealType>("Bulk Modulus", 0.0);
  shear_modulus_ = p->get<RealType>("Shear Modulus", 0.0);

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Strain", dl->qp_tensor));

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

//
//
//
template <typename EvalT, typename Traits>
void
LinearElasticVolDevModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // constants
  auto const kappa = bulk_modulus_;
  auto const mu    = shear_modulus_;

  // extract dependent MDFields
  auto strain = *dep_fields["Strain"];

  // extract evaluated MDFields
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];
  auto        stress = *eval_fields[cauchy];

  minitensor::Tensor<ScalarT> eps(num_dims_), sigma(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      eps.fill(strain, cell, pt, 0, 0);

      auto const sigma_vol = num_dims_ * kappa * minitensor::vol(eps);
      auto const sigma_dev = 2.0 * mu * minitensor::dev(eps);

      sigma = sigma_vol + sigma_dev;

      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }
}
}  // namespace LCM
