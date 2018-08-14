//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include <MiniTensor.h>

#include <typeinfo>

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
TransportCoefficients<EvalT, Traits>::TransportCoefficients(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : c_lattice_(
          p.get<std::string>("Lattice Concentration Name"),
          dl->qp_scalar),
      temperature_(p.get<std::string>("Temperature Name"), dl->qp_scalar),
      k_eq_(
          p.get<std::string>("Concentration Equilibrium Parameter Name"),
          dl->qp_scalar),
      n_trap_(p.get<std::string>("Trapped Solvent Name"), dl->qp_scalar),
      c_trapped_(
          p.get<std::string>("Trapped Concentration Name"),
          dl->qp_scalar),
      eff_diff_(
          p.get<std::string>("Effective Diffusivity Name"),
          dl->qp_scalar),
      diffusion_coefficient_(
          p.get<std::string>("Diffusion Coefficient Name"),
          dl->qp_scalar),
      convection_coefficient_(
          p.get<std::string>("Tau Contribution Name"),
          dl->qp_scalar),
      total_concentration_(
          p.get<std::string>("Total Concentration Name"),
          dl->qp_scalar),
      F_(p.get<std::string>("Deformation Gradient Name"), dl->qp_tensor),
      F_mech_(
          p.get<std::string>("Mechanical Deformation Gradient Name"),
          dl->qp_tensor),
      J_(p.get<std::string>("Determinant of F Name"), dl->qp_scalar),
      // strain_rate_fac_(p.get<std::string>("Strain Rate Factor
      // Name"),dl->qp_scalar),
      weighted_average_(p.get<bool>("Weighted Volume Average J", false)),
      alpha_(p.get<RealType>("Average J Stabilization Parameter", 0.0))
{
  field_name_map_ =
      p.get<Teuchos::RCP<std::map<std::string, std::string>>>("Name Map");

  // get the material parameter list
  Teuchos::ParameterList* mat_params =
      p.get<Teuchos::ParameterList*>("Material Parameters");

  partial_molar_volume_   = mat_params->get<RealType>("Partial Molar Volume");
  pre_exponential_factor_ = mat_params->get<RealType>("Pre-exponential Factor");
  Q_ = mat_params->get<RealType>("Diffusion Activation Enthalpy");
  ideal_gas_constant_  = mat_params->get<RealType>("Ideal Gas Constant");
  trap_binding_energy_ = mat_params->get<RealType>("Trap Binding Energy");
  n_lattice_           = mat_params->get<RealType>("Number of Lattice Sites");
  ref_total_concentration_ =
      mat_params->get<RealType>("Reference Total Concentration");
  a_                   = mat_params->get<RealType>("A Constant");
  b_                   = mat_params->get<RealType>("B Constant");
  c_                   = mat_params->get<RealType>("C Constant");
  avogadros_num_       = mat_params->get<RealType>("Avogadro's Number");
  lattice_strain_flag_ = mat_params->get<bool>("Lattice Strain Flag");

  have_eqps_ = false;
  if (p.isType<std::string>("Equivalent Plastic Strain Name")) {
    have_eqps_       = true;
    strain_rate_fac_ = decltype(strain_rate_fac_)(
        p.get<std::string>("Strain Rate Factor Name"), dl->qp_scalar);
    this->addEvaluatedField(strain_rate_fac_);
  }

  this->addDependentField(F_);
  this->addDependentField(J_);
  this->addDependentField(temperature_);
  this->addDependentField(c_lattice_);
  this->addEvaluatedField(k_eq_);
  this->addEvaluatedField(n_trap_);
  this->addEvaluatedField(eff_diff_);
  this->addEvaluatedField(c_trapped_);
  this->addEvaluatedField(total_concentration_);
  this->addEvaluatedField(diffusion_coefficient_);
  this->addEvaluatedField(convection_coefficient_);
  this->addEvaluatedField(F_mech_);

  this->setName("Transport Coefficients" + PHX::typeAsString<EvalT>());
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  worksetSize = dims[0];
  num_pts_    = dims[1];
  num_dims_   = dims[2];
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
TransportCoefficients<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(temperature_, fm);
  this->utils.setFieldData(c_lattice_, fm);
  this->utils.setFieldData(F_, fm);
  this->utils.setFieldData(F_mech_, fm);
  this->utils.setFieldData(J_, fm);

  this->utils.setFieldData(k_eq_, fm);
  this->utils.setFieldData(c_trapped_, fm);
  this->utils.setFieldData(total_concentration_, fm);
  this->utils.setFieldData(n_trap_, fm);
  this->utils.setFieldData(eff_diff_, fm);
  if (have_eqps_) this->utils.setFieldData(strain_rate_fac_, fm);
  this->utils.setFieldData(diffusion_coefficient_, fm);
  this->utils.setFieldData(convection_coefficient_, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
TransportCoefficients<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // std::cout << "In evaluator: " << this->getName() << "\n";

  ScalarT theta_term(0.0);

  // Use the previous iterate of eqps to avoid circular dependency issue
  Albany::MDArray eqps;
  if (have_eqps_) {
    std::string eqps_string = (*field_name_map_)["eqps"];
    eqps                    = (*workset.stateArrayPtr)[eqps_string + "_old"];
  }

  // Diffusion Coefficient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      diffusion_coefficient_(cell, pt) =
          pre_exponential_factor_ *
          std::exp(-1.0 * Q_ / (ideal_gas_constant_ * temperature_(cell, pt)));
    }
  }

  // Convection Coefficient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      convection_coefficient_(cell, pt) =
          partial_molar_volume_ * diffusion_coefficient_(cell, pt) /
          (ideal_gas_constant_ * temperature_(cell, pt));
    }
  }

  // equilibrium constant k_T = e^(W_b/RT)
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      k_eq_(cell, pt) = std::exp(
          trap_binding_energy_ /
          (ideal_gas_constant_ * temperature_(cell, pt)));
      //   	std::cout  << "k_eq_" << k_eq_(cell,pt) << std::endl;
    }
  }

  /*
  // theta term C_T
  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      theta_term = k_eq_(cell,pt) * c_lattice_(cell,pt) /
        ( k_eq_(cell,pt) * c_lattice_(cell,pt) + n_lattice_ );
    }
  }
  */

  // trapped solvent
  if (have_eqps_) {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        n_trap_(cell, pt) =
            (1.0 / avogadros_num_) *
            std::pow(10.0, (a_ - b_ * std::exp(-c_ * eqps(cell, pt))));
        //     std::cout  << "ntrap" << n_trap_(cell,pt) << std::endl;
      }
    }
  } else {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        n_trap_(cell, pt) = (1.0 / avogadros_num_) * std::pow(10.0, (a_ - b_));
      }
    }
  }

  // strain rate factor
  if (have_eqps_) {
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        theta_term = k_eq_(cell, pt) * c_lattice_(cell, pt) /
                     (k_eq_(cell, pt) * c_lattice_(cell, pt) + n_lattice_);

        strain_rate_fac_(cell, pt) = theta_term * n_trap_(cell, pt) *
                                     std::log(10.0) * b_ * c_ *
                                     std::exp(-c_ * eqps(cell, pt));
      }
    }
  }
  // else
  // {
  //   for (std::size_t cell(0); cell < workset.numCells; ++cell) {
  //     for (std::size_t pt(0); pt < num_pts_; ++pt) {
  //       theta_term = k_eq_(cell,pt) * c_lattice_(cell,pt) /
  //         ( k_eq_(cell,pt) * c_lattice_(cell,pt) + n_lattice_ );

  //       strain_rate_fac_(cell,pt) = theta_term * n_trap_(cell,pt) *
  //         std::log(10.0) * b_ * c_;
  //     }
  //   }
  // }

  // trapped concentration
  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      theta_term = k_eq_(cell, pt) * c_lattice_(cell, pt) /
                   (k_eq_(cell, pt) * c_lattice_(cell, pt) + n_lattice_);

      c_trapped_(cell, pt) = theta_term * n_trap_(cell, pt);
    }
  }

  // total concentration
  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      total_concentration_(cell, pt) =
          c_trapped_(cell, pt) + c_lattice_(cell, pt);
    }
  }

  // effective diffusivity
  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      eff_diff_(cell, pt) =
          1.0 +
          n_trap_(cell, pt) * n_lattice_ /
              (k_eq_(cell, pt) * c_lattice_(cell, pt) * c_lattice_(cell, pt)) /
              ((1.0 + n_lattice_ / k_eq_(cell, pt) / c_lattice_(cell, pt)) *
               (1.0 + n_lattice_ / k_eq_(cell, pt) / c_lattice_(cell, pt)));
    }
  }

  // deformation gradient volumetric split for lattice concentration
  minitensor::Tensor<ScalarT> Fmech(num_dims_);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      Fmech.fill(F_, cell, pt, 0, 0);
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          F_mech_(cell, pt, i, j) = Fmech(i, j);
        }
      }
    }
  }

  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable
  // values. Leaving this out leads to inversion of 0 tensors.
  for (int cell = workset.numCells; cell < worksetSize; ++cell)
    for (int qp = 0; qp < num_pts_; ++qp)
      for (int i = 0; i < num_dims_; ++i) F_mech_(cell, qp, i, i) = 1.0;

  ScalarT lambda_ = partial_molar_volume_ * n_lattice_ / avogadros_num_;
  ScalarT JH(1.0);

  if (lattice_strain_flag_) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_pts_; ++qp) {
        JH = 1.0 + lambda_ * (total_concentration_(cell, qp) -
                              ref_total_concentration_);
        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            F_mech_(cell, qp, i, j) *= std::pow(JH, -1. / 3.);
          }
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace LCM
