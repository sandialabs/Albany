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
ThermoMechanicalCoefficients<EvalT, Traits>::ThermoMechanicalCoefficients(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thermal_cond_(
          p.get<std::string>("Thermal Conductivity Name"),
          dl->qp_scalar),
      thermal_transient_coeff_(
          p.get<std::string>("Thermal Transient Coefficient Name"),
          dl->qp_scalar),
      thermal_diffusivity_(
          p.get<std::string>("Thermal Diffusivity Name"),
          dl->qp_tensor),
      SolutionType_(p.get<std::string>("Solution Method Type")),
      have_mech_(p.get<bool>("Have Mechanics", false))
{
  // get the material parameter list
  Teuchos::ParameterList* mat_params =
      p.get<Teuchos::ParameterList*>("Material Parameters");

  transient_coeff_ = mat_params->get<RealType>("Thermal Transient Coefficient");
  heat_capacity_   = mat_params->get<RealType>("Heat Capacity");
  density_         = mat_params->get<RealType>("Density");

  if (SolutionType_ == "Continuation") {
    temperature_ = decltype(temperature_)(
        p.get<std::string>("Temperature Name"), dl->qp_scalar);

    temperature_dot_ = decltype(temperature_dot_)(
        p.get<std::string>("Temperature Dot Name"), dl->qp_scalar);

    delta_time_ = decltype(delta_time_)(
        p.get<std::string>("Delta Time Name"), dl->workset_scalar);

    this->addDependentField(temperature_);
    this->addDependentField(delta_time_);
    this->addEvaluatedField(temperature_dot_);
  }
  this->addDependentField(thermal_cond_);
  this->addEvaluatedField(thermal_transient_coeff_);
  this->addEvaluatedField(thermal_diffusivity_);

  this->setName("ThermoMechanical Coefficients" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;

  dl->qp_tensor->dimensions(dims);

  num_pts_  = dims[1];
  num_dims_ = dims[2];

  if (have_mech_) {
    def_grad_ = decltype(def_grad_)(
        p.get<std::string>("Deformation Gradient Name"), dl->qp_tensor);
    this->addDependentField(def_grad_);
  }

  temperature_name_ = p.get<std::string>("Temperature Name") + "_old";
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ThermoMechanicalCoefficients<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  if (SolutionType_ == "Continuation") {
    this->utils.setFieldData(temperature_, fm);
    this->utils.setFieldData(temperature_dot_, fm);
    this->utils.setFieldData(delta_time_, fm);
  }

  this->utils.setFieldData(thermal_cond_, fm);
  this->utils.setFieldData(thermal_transient_coeff_, fm);
  this->utils.setFieldData(thermal_diffusivity_, fm);

  if (have_mech_) { this->utils.setFieldData(def_grad_, fm); }
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ThermoMechanicalCoefficients<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (SolutionType_ == "Continuation") {
    Albany::MDArray const temperature_old =
        (*workset.stateArrayPtr)[temperature_name_];

    ScalarT dt = delta_time_(0);

    if (dt == 0.0) {
      // Initially, transfer the derivatives of temperature_ to the
      // derivatives of temperature_dot_
      dt = 1.0;
    }

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        temperature_dot_(cell, pt) =
            (temperature_(cell, pt) - temperature_old(cell, pt)) / dt;
      }
    }
  }

  if (have_mech_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        minitensor::Tensor<ScalarT> F(num_dims_);

        F.fill(def_grad_, cell, pt, 0, 0);

        minitensor::Tensor<ScalarT> tensor =
            minitensor::inverse(minitensor::transpose(F) * F);

        thermal_transient_coeff_(cell, pt) = transient_coeff_;

        minitensor::Tensor<ScalarT> diffusivity =
            thermal_cond_(cell, pt) / (density_ * heat_capacity_) * tensor;

        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            thermal_diffusivity_(cell, pt, i, j) = diffusivity(i, j);
          }
        }
      }
    }
  } else {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        thermal_transient_coeff_(cell, pt) = transient_coeff_;

        minitensor::Tensor<RealType> I(minitensor::eye<RealType>(num_dims_));

        minitensor::Tensor<ScalarT> diffusivity =
            thermal_cond_(cell, pt) / (density_ * heat_capacity_) * I;

        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            thermal_diffusivity_(cell, pt, i, j) = diffusivity(i, j);
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
