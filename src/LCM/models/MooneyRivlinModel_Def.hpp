//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid2_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
MooneyRivlinModel<EvalT, Traits>::
MooneyRivlinModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
        c1_(p->get<RealType>("c1", 0.0)),
        c2_(p->get<RealType>("c2", 0.0)),
        c_(p->get<RealType>("c", 0.0))
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("F", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("J", dl->qp_scalar));

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
template<typename EvalT, typename Traits>
void MooneyRivlinModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
{
  // extract dependent MDFields
  PHX::MDField<ScalarT> defGrad = *dep_fields["F"];
  PHX::MDField<ScalarT> J = *dep_fields["J"];
  // extract evaluated MDFields
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy];

  Intrepid2::Tensor<ScalarT> F(num_dims_), C(num_dims_);
  Intrepid2::Tensor<ScalarT> S(num_dims_), sigma(num_dims_);
  Intrepid2::Tensor<ScalarT> I(Intrepid2::eye<ScalarT>(num_dims_));

  ScalarT d = 2.0 * (c1_ + 2 * c2_);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      F.fill(defGrad,cell, pt,0,0);
      C = transpose(F) * F;
      S = 2.0 * (c1_ + c2_ * Intrepid2::I1(C)) * I - 2.0 * c2_ * C
          + (2.0 * c_ * J(cell, pt) * (J(cell, pt) - 1.0) - d)
              * Intrepid2::inverse(C);
      sigma = (1. / J(cell, pt)) * F * S * Intrepid2::transpose(F);

      for (int i(0); i < num_dims_; ++i)
        for (int j(0); j < num_dims_; ++j)
          stress(cell, pt, i, j) = sigma(i, j);
    }
  }
}
//------------------------------------------------------------------------------
}

