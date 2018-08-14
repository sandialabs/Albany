//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
StabilizedPressureResidual<EvalT, Traits>::StabilizedPressureResidual(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : shear_modulus_(p.get<std::string>("Shear Modulus Name"), dl->qp_scalar),
      bulk_modulus_(p.get<std::string>("Bulk Modulus Name"), dl->qp_scalar),
      def_grad_(p.get<std::string>("Deformation Gradient Name"), dl->qp_tensor),
      stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),
      pressure_(p.get<std::string>("Pressure Name"), dl->qp_scalar),
      pressure_grad_(
          p.get<std::string>("Pressure Gradient Name"),
          dl->qp_vector),
      w_grad_bf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      h_(p.get<std::string>("Element Characteristic Length Name"),
         dl->qp_scalar),
      residual_(p.get<std::string>("Residual Name"), dl->node_scalar),
      small_strain_(p.get<bool>("Small Strain", false)),
      alpha_(p.get<RealType>("Stabilization Parameter"))
{
  this->addDependentField(shear_modulus_);
  this->addDependentField(bulk_modulus_);
  this->addDependentField(def_grad_);
  this->addDependentField(stress_);
  this->addDependentField(pressure_);
  this->addDependentField(pressure_grad_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(w_bf_);
  this->addDependentField(h_);

  this->addEvaluatedField(residual_);

  this->setName("StabilizedPressureResidual" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
StabilizedPressureResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(shear_modulus_, fm);
  this->utils.setFieldData(bulk_modulus_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(pressure_, fm);
  this->utils.setFieldData(pressure_grad_, fm);
  this->utils.setFieldData(w_grad_bf_, fm);
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(h_, fm);
  this->utils.setFieldData(residual_, fm);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
StabilizedPressureResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  minitensor::Tensor<ScalarT> sigma(num_dims_);

  if (small_strain_) {
    // small strain version needs no pull back
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes_; ++node) {
        residual_(cell, node) = 0.0;
      }
      for (int pt = 0; pt < num_pts_; ++pt) {
        sigma.fill(stress_, cell, pt, 0, 0);
        ScalarT dUdJ = (1.0 / num_dims_) * minitensor::trace(sigma);
        for (int node = 0; node < num_nodes_; ++node) {
          residual_(cell, node) += w_bf_(cell, node, pt) *
                                   (dUdJ - pressure_(cell, pt)) /
                                   bulk_modulus_(cell, pt);
        }
      }

      // stabilization term
      for (int pt = 0; pt < num_pts_; ++pt) {
        ScalarT stab_term  = 0.5 * alpha_ * h_(cell, pt) * h_(cell, pt);
        ScalarT stab_param = stab_term / shear_modulus_(cell, pt);
        for (int node = 0; node < num_nodes_; ++node) {
          for (int i = 0; i < num_dims_; ++i) {
            residual_(cell, node) -= stab_param *
                                     w_grad_bf_(cell, node, pt, i) *
                                     pressure_grad_(cell, pt, i);
          }
        }
      }
    }
  } else {
    minitensor::Tensor<ScalarT> F(num_dims_);
    minitensor::Tensor<ScalarT> Cinv(num_dims_);

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes_; ++node) {
        residual_(cell, node) = 0.0;
      }
      for (int pt = 0; pt < num_pts_; ++pt) {
        sigma.fill(stress_, cell, pt, 0, 0);
        ScalarT dUdJ = (1.0 / num_dims_) * minitensor::trace(sigma);
        for (int node = 0; node < num_nodes_; ++node) {
          residual_(cell, node) += w_bf_(cell, node, pt) *
                                   (dUdJ - pressure_(cell, pt)) /
                                   bulk_modulus_(cell, pt);
        }
      }

      // stabilization term
      ScalarT stab_term = 0.5 * alpha_;
      for (int pt = 0; pt < num_pts_; ++pt) {
        F.fill(def_grad_, cell, pt, 0, 0);
        ScalarT J = minitensor::det(F);
        Cinv      = minitensor::inverse(minitensor::transpose(F) * F);
        ScalarT stab_param =
            stab_term * h_(cell, pt) * h_(cell, pt) / shear_modulus_(cell, pt);
        for (int node = 0; node < num_nodes_; ++node) {
          for (int i = 0; i < num_dims_; ++i) {
            for (int j = 0; j < num_dims_; ++j) {
              residual_(cell, node) -= stab_param * J * Cinv(i, j) *
                                       pressure_grad_(cell, pt, i) *
                                       w_grad_bf_(cell, node, pt, j);
            }
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
