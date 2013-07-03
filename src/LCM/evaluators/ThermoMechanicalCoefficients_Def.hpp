//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Intrepid_MiniTensor.h>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  ThermoMechanicalCoefficients<EvalT, Traits>::
  ThermoMechanicalCoefficients(Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl) :
    temperature_(p.get<std::string>("Temperature Name"),dl->qp_scalar),
    thermal_cond_(p.get<std::string>("Thermal Conductivity Name"),dl->qp_scalar),
    thermal_transient_coeff_(p.get<std::string>("Thermal Transient Coefficient Name"),dl->qp_scalar),
    thermal_diffusivity_(p.get<std::string>("Thermal Diffusivity Name"),dl->qp_tensor),
    have_mech_(p.get<bool>("Have Mechanics", false))
  {
    // get the material parameter list
    Teuchos::ParameterList* mat_params = 
      p.get<Teuchos::ParameterList*>("Material Parameters");

    transient_coeff_ = mat_params->get<RealType>("Thermal Transient Coefficient");
    expansion_coeff_ = mat_params->get<RealType>("Thermal Expansion Coefficient");
    heat_capacity_   = mat_params->get<RealType>("Heat Capacity");
    density_         = mat_params->get<RealType>("Density");
    ref_temperature_ = mat_params->get<RealType>("Reference Temperature");

    this->addDependentField(temperature_);
    this->addDependentField(thermal_cond_);
 
    this->addEvaluatedField(thermal_transient_coeff_);
    this->addEvaluatedField(thermal_diffusivity_);

    this->setName("ThermoMechanical Coefficients"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];

    if (have_mech_) {
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim>
        temp_def_grad(p.get<string>("Deformation Gradient Name"), dl->qp_tensor);
      def_grad_ = temp_def_grad;
      this->addDependentField(def_grad_);

      PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim>
        temp_stress(p.get<string>("Stress Name"), dl->qp_tensor);
      stress_ = temp_stress;
      this->addDependentField(stress_);
      this->addEvaluatedField(stress_);

      PHX::MDField<ScalarT,Cell,QuadPoint>
        temp_source(p.get<string>("Mechanical Source Name"), dl->qp_scalar);
      source_ = temp_source;
      this->addDependentField(source_);
      this->addEvaluatedField(source_);
    }
      
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ThermoMechanicalCoefficients<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(temperature_,fm);
    this->utils.setFieldData(thermal_cond_,fm);
    this->utils.setFieldData(thermal_transient_coeff_,fm);
    this->utils.setFieldData(thermal_diffusivity_,fm);
    if (have_mech_) {
      this->utils.setFieldData(def_grad_,fm);
      this->utils.setFieldData(stress_,fm);
      this->utils.setFieldData(source_,fm);
    }
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ThermoMechanicalCoefficients<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    Intrepid::Tensor<ScalarT> diffusivity(num_dims_);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
    Intrepid::Tensor<ScalarT> tensor = I;
    Intrepid::Tensor<ScalarT> F(num_dims_), sigma(num_dims_);
    ScalarT J = 1.0;
    
    if (have_mech_) {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t pt = 0; pt < num_pts_; ++pt) {
          F.fill( &def_grad_(cell,pt,0,0) );
          J = Intrepid::det(F);
          sigma.fill( &stress_(cell,pt,0,0) );
          tensor = Intrepid::inverse(Intrepid::transpose(F)*F);
          source_(cell,pt) = source_(cell,pt) / (density_ * heat_capacity_);
          sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J*J))
            * (temperature_(cell,pt) - ref_temperature_) * I;
        }
      }
    }

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_pts_; ++pt) {
        thermal_transient_coeff_(cell,pt) = transient_coeff_;
        diffusivity = thermal_cond_(cell,pt) / (density_ * heat_capacity_) * tensor;
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            thermal_diffusivity_(cell,pt,i,j) = diffusivity(i,j);
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
}

