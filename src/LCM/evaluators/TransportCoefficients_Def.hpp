//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  TransportCoefficients<EvalT, Traits>::
  TransportCoefficients(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    c_lattice_(p.get<std::string>("Lattice Concentration Name"),dl->qp_scalar),
    temperature_(p.get<std::string>("Temperature Name"),dl->qp_scalar),
    k_eq_(p.get<std::string>("Concentration Equilibrium Parameter Name"),dl->qp_scalar),
    n_trap_(p.get<std::string>("Trapped Solvent Name"),dl->qp_scalar),
    c_trapped_(p.get<std::string>("Trapped Concentration Name"),dl->qp_scalar),
    eff_diff_(p.get<std::string>("Effective Diffusivity Name"),dl->qp_scalar),
    diffusion_coefficient_(p.get<std::string>("Diffusion Coefficient Name"),dl->qp_scalar),
    convection_coefficient_(p.get<std::string>("Tau Contribution Name"),dl->qp_scalar),
    total_concentration_(p.get<std::string>("Total Concentration Name"),dl->qp_scalar),
    strain_rate_fac_(p.get<std::string>("Strain Rate Factor Name"),dl->qp_scalar)
  {
    // get the material parameter list
    Teuchos::ParameterList* mat_params = 
      p.get<Teuchos::ParameterList*>("Material Parameters");

    partial_molar_volume_ = mat_params->get<RealType>("Partial Molar Volume");
    pre_exponential_factor_ = mat_params->get<RealType>("Pre-exponential Factor");
    Q_ = mat_params->get<RealType>("Diffusion Activation Enthalpy");
    ideal_gas_constant_ = mat_params->get<RealType>("Ideal Gas Constant");
	trap_binding_energy_ = mat_params->get<RealType>("Trap Binding Energy");
    n_lattice_ = mat_params->get<RealType>("Number of Lattice Sites");
    a_ = mat_params->get<RealType>("A Constant");
    b_ = mat_params->get<RealType>("B Constant");
    c_ = mat_params->get<RealType>("C Constant");
    avogadros_num_ = 6.02214e23;

    have_eqps_ = false;
    if ( p.isType<std::string>("Equivalent Plastic Strain Name") ) {
      have_eqps_ = true;
      PHX::MDField<ScalarT, Cell, QuadPoint> 
        tmp(p.get<std::string>("Equivalent Plastic Strain Name"), dl->qp_scalar);
      eqps_ = tmp;
    }

    this->addDependentField(temperature_);
    this->addDependentField(c_lattice_);
    if (have_eqps_) {
      this->addDependentField(eqps_);
    }
    this->addEvaluatedField(k_eq_);
    this->addEvaluatedField(n_trap_);
    this->addEvaluatedField(eff_diff_);
    this->addEvaluatedField(c_trapped_);
    this->addEvaluatedField(total_concentration_);
    this->addEvaluatedField(strain_rate_fac_);
    this->addEvaluatedField(diffusion_coefficient_);
    this->addEvaluatedField(convection_coefficient_);

    this->setName("Transport Coefficients"+PHX::TypeString<EvalT>::value);
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_scalar->dimensions(dims);
    num_pts_ = dims[1];

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void TransportCoefficients<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
	this->utils.setFieldData(temperature_,fm);
    this->utils.setFieldData(c_lattice_,fm);
    if ( have_eqps_ ) {
      this->utils.setFieldData(eqps_,fm);
    }
    this->utils.setFieldData(k_eq_,fm);
    this->utils.setFieldData(c_trapped_,fm);
    this->utils.setFieldData(total_concentration_,fm);
    this->utils.setFieldData(n_trap_,fm);
    this->utils.setFieldData(eff_diff_,fm);
    this->utils.setFieldData(strain_rate_fac_,fm);
    this->utils.setFieldData(diffusion_coefficient_,fm);
    this->utils.setFieldData(convection_coefficient_,fm);

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void TransportCoefficients<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    ScalarT theta_term(0.0);

    // Diffusion Coefficient
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < num_pts_; ++pt) {

        diffusion_coefficient_(cell,pt) = pre_exponential_factor_*
          			                                                       std::exp(-1.0*Q_/
          			                                        		ideal_gas_constant_/
          			                                        		temperature_(cell,pt));
      }
    }

    // Convection Coefficient
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < num_pts_; ++pt) {

        convection_coefficient_(cell,pt) = partial_molar_volume_*
        	                                            	diffusion_coefficient_(cell,pt)/
          			                                        		(ideal_gas_constant_*
          			                                        		temperature_(cell,pt));
      }
    }

    // equilibrium constant k_T = e^(W_b/RT)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < num_pts_; ++pt) {

      	k_eq_(cell,pt) = std::exp(trap_binding_energy_/
      			                                            ideal_gas_constant_/
      			                                            temperature_(cell,pt));
      }
    }

    // theta term C_T
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        theta_term = k_eq_(cell,pt) * c_lattice_(cell,pt) / 
          ( k_eq_(cell,pt) * c_lattice_(cell,pt) + n_lattice_ );
      }
    }
    
    // trapped solvent
    if (have_eqps_) {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          n_trap_(cell,pt) = (1.0/avogadros_num_) * 
            std::pow( 10.0, a_ - b_ * std::exp( -c_ * eqps_(cell,pt) ) );
        }
      }
    }
    else
    {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          n_trap_(cell,pt) = (1.0/avogadros_num_) * std::pow( 10.0, a_ - b_ );
        }
      }
    }
    
    // strain rate factor
    if (have_eqps_) {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          strain_rate_fac_(cell,pt) = theta_term * n_trap_(cell,pt) * 
            std::log(10.0) * b_ * c_ * std::exp( -c_ * eqps_(cell,pt) );
        }
      }
    }
    else
    {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          strain_rate_fac_(cell,pt) = theta_term * n_trap_(cell,pt) * 
            std::log(10.0) * b_ * c_;
        }
      }
    }

    // trapped conecentration
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        c_trapped_(cell,pt) = theta_term * n_trap_(cell,pt);
      }
    }
    
    // total concentration
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
    	  total_concentration_(cell, pt) = c_trapped_(cell,pt) +
    			                                               c_lattice_(cell,pt);
      }
    }

    // effective diffusivity
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        eff_diff_(cell,pt) = 1.0 + n_trap_(cell,pt) * n_lattice_ /
          (  k_eq_(cell,pt) * c_lattice_(cell,pt) * c_lattice_(cell,pt) ) /
          ( ( 1.0 + n_lattice_ / k_eq_(cell,pt) / c_lattice_(cell,pt) ) *
            ( 1.0 + n_lattice_ / k_eq_(cell,pt) / c_lattice_(cell,pt) ) );
      }
    }

  }
  //----------------------------------------------------------------------------
}

