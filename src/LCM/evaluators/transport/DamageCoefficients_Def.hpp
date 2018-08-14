//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
DamageCoefficients<EvalT, Traits>::DamageCoefficients(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : damage_(p.get<std::string>("Damage Name"), dl->qp_scalar),
      delta_time_(p.get<std::string>("Delta Time Name"), dl->workset_scalar),
      damage_transient_coeff_(
          p.get<std::string>("Damage Transient Coefficient Name"),
          dl->qp_scalar),
      damage_diffusivity_(
          p.get<std::string>("Damage Diffusivity Name"),
          dl->qp_tensor),
      damage_dot_(p.get<std::string>("Damage Dot Name"), dl->qp_scalar),
      have_mech_(p.get<bool>("Have Mechanics", false))
{
  // get the material parameter list
  Teuchos::ParameterList* mat_params =
      p.get<Teuchos::ParameterList*>("Material Parameters");

  transient_coeff_ = mat_params->get<RealType>("Damage Transient Coefficient");
  diffusivity_coeff_ =
      mat_params->get<RealType>("Damage Diffusivity Coefficient");

  this->addDependentField(damage_);
  this->addDependentField(delta_time_);

  this->addEvaluatedField(damage_transient_coeff_);
  this->addEvaluatedField(damage_diffusivity_);
  this->addEvaluatedField(damage_dot_);

  this->setName("Damage Coefficients" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  num_pts_  = dims[1];
  num_dims_ = dims[2];

  if (have_mech_) {
    def_grad_ = decltype(def_grad_)(
        p.get<std::string>("Deformation Gradient Name"), dl->qp_tensor);
    this->addDependentField(def_grad_);
  }
  damage_name_ = p.get<std::string>("Damage Name") + "_old";
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
DamageCoefficients<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(damage_, fm);
  this->utils.setFieldData(damage_dot_, fm);
  this->utils.setFieldData(delta_time_, fm);
  this->utils.setFieldData(damage_transient_coeff_, fm);
  this->utils.setFieldData(damage_diffusivity_, fm);
  if (have_mech_) { this->utils.setFieldData(def_grad_, fm); }
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
DamageCoefficients<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  minitensor::Tensor<ScalarT> diffusivity(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor<ScalarT> tensor;
  minitensor::Tensor<ScalarT> F(num_dims_);

  ScalarT dt = delta_time_(0);
  if (dt == 0.0) dt = 1.e-15;
  Albany::MDArray damage_old = (*workset.stateArrayPtr)[damage_name_];
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      damage_dot_(cell, pt) = (damage_(cell, pt) - damage_old(cell, pt)) / dt;
    }
  }

  if (have_mech_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        F.fill(def_grad_, cell, pt, 0, 0);
        tensor = minitensor::inverse(minitensor::transpose(F) * F);
        damage_transient_coeff_(cell, pt) = transient_coeff_;
        diffusivity                       = diffusivity_coeff_ * tensor;
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            damage_diffusivity_(cell, pt, i, j) = diffusivity(i, j);
          }
        }
      }
    }
  } else {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        damage_transient_coeff_(cell, pt) = transient_coeff_;
        diffusivity                       = diffusivity_coeff_ * I;
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            damage_diffusivity_(cell, pt, i, j) = diffusivity(i, j);
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
