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
NewtonianFluidModel<EvalT, Traits>::NewtonianFluidModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      mu_(p->get<RealType>("Shear Viscosity", 1.0))
{
  // retrive appropriate field name strings
  std::string F_string      = (*field_name_map_)["F"];
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));

  // define the state variables
  //
  // F
  this->num_state_variables_++;
  this->state_var_names_.push_back(F_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output F", false));
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
NewtonianFluidModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  std::string F_string      = (*field_name_map_)["F"];
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];

  // extract dependent MDFields
  auto def_grad   = *dep_fields[F_string];
  auto delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  auto stress = *eval_fields[cauchy_string];

  // get State Variables
  Albany::MDArray def_grad_old = (*workset.stateArrayPtr)[F_string + "_old"];

  // pressure is hard coded as 1 for now
  // this is likely not general enough :)
  ScalarT p = 1;

  // time increment
  ScalarT dt = delta_time(0);

  // containers
  minitensor::Tensor<ScalarT> Fnew(num_dims_);
  minitensor::Tensor<ScalarT> Fold(num_dims_);
  minitensor::Tensor<ScalarT> Finc(num_dims_);
  minitensor::Tensor<ScalarT> L(num_dims_);
  minitensor::Tensor<ScalarT> D(num_dims_);
  minitensor::Tensor<ScalarT> sigma(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      // should only be the first time step
      if (dt == 0) {
        for (int i = 0; i < num_dims_; ++i)
          for (int j = 0; j < num_dims_; ++j) stress(cell, pt, i, j) = 0.0;
      } else {
        // old deformation gradient
        for (int i = 0; i < num_dims_; ++i)
          for (int j = 0; j < num_dims_; ++j)
            Fold(i, j) = ScalarT(def_grad_old(cell, pt, i, j));

        // current deformation gradient
        Fnew.fill(def_grad, cell, pt, 0, 0);

        // incremental deformation gradient
        Finc = Fnew * minitensor::inverse(Fold);

        // velocity gradient
        L = (1.0 / dt) * minitensor::log(Finc);

        // strain rate (a.k.a rate of deformation)
        D = minitensor::sym(L);

        // stress tensor
        sigma =
            -p * I + 2.0 * mu_ * (D - (2.0 / 3.0) * minitensor::trace(D) * I);

        // update stress state
        for (int i = 0; i < num_dims_; ++i)
          for (int j = 0; j < num_dims_; ++j)
            stress(cell, pt, i, j) = sigma(i, j);
      }
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
