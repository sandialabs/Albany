//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>
#include "Albany_Utils.hpp"

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
CP::SlipFamily<NumDimT, NumSlipT>::SlipFamily()
{

}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::SlipFamily<NumDimT, NumSlipT>::setHardeningLawType(CP::HardeningLawType law)
{
  type_hardening_law_ = law;

  phardening_parameters_ =
    CP::hardeningParameterFactory<CP::MAX_DIM, CP::MAX_SLIP>(type_hardening_law_);
}

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
void
CP::SlipFamily<NumDimT, NumSlipT>::setFlowRuleType(CP::FlowRuleType rule)
{
  type_flow_rule_ = rule;

  pflow_parameters_ = CP::flowParameterFactory(type_flow_rule_);
}


//
// Verify that constitutive update has preserved finite values
//
template<typename T, minitensor::Index N>
void
CP::expectFiniteTensor(
    minitensor::Tensor<T, N> const & A,
    std::string const & msg)
{
  minitensor::Index const
  dim = A.get_dimension();

  minitensor::Index const
  num_components = dim * dim;

  for (minitensor::Index i = 0; i < num_components; ++i) {
    ALBANY_EXPECT(boost::math::isfinite(Sacado::ScalarValue<T>::eval(A[i])) == true);
  }
}

///
/// Update the plastic quantities
///
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::applySlipIncrement(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    minitensor::Vector<RealType, NumSlipT> const & slip_n,
    minitensor::Vector<ArgT, NumSlipT> const & slip_np1,
    minitensor::Tensor<RealType, NumDimT> const & Fp_n,
    minitensor::Tensor<ArgT, NumDimT> & Lp_np1,
    minitensor::Tensor<ArgT, NumDimT> & Fp_np1)
{
  minitensor::Index const
  num_slip = slip_n.get_dimension();

  minitensor::Index const
  num_dim = Fp_n.get_dimension();

  //
  // calculate plastic velocity gradient
  //
  Lp_np1.fill(minitensor::Filler::ZEROS);
  Lp_np1 = 0. * Lp_np1;

  if(dt > 0){
    for (minitensor::Index s(0); s < num_slip; ++s) {
      for (int i = 0; i < num_dim; ++i) {
        for (int j = 0; j < num_dim; ++j) {
          Lp_np1(i,j) += (slip_np1[s] - slip_n[s])/dt * slip_systems.at(s).projector_(i,j);
        }
      }
    }
  }

  CP::expectFiniteTensor(Lp_np1, "Lp_np1 in applySlipIncrement().");

  // update plastic deformation gradient
  // F^{p}_{n+1} = exp(L_{n+1} * delta t) F^{p}_{n}
  minitensor::Tensor<ArgT, NumDimT> const
  exp_L_dt = minitensor::exp(Lp_np1 * dt);

  Fp_np1 = exp_L_dt * Fp_n;

  CP::expectFiniteTensor(Fp_np1, "Fp_np1 in applySlipIncrement()");

  if (minitensor::det(Fp_np1) == 0.0)
  {
    std::cout << "Singular plastic deformation gradient" << std::endl;
    std::cout << std::setprecision(4) << Fp_np1 << std::endl;
    // failed = true;
    return;
  }
}


///
/// Evolve the hardnesses
///
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::updateHardness(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    minitensor::Vector<ArgT, NumSlipT> const & rate_slip,
    minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
    minitensor::Vector<ArgT, NumSlipT> & state_hardening_np1,
    minitensor::Vector<ArgT, NumSlipT> & slip_resistance,
    bool & failed)
{
  for (unsigned int sf_index(0); sf_index < slip_families.size(); ++ sf_index)
  {
    auto const &
    slip_family = slip_families[sf_index];

    auto
    type_hardening_law = slip_family.getHardeningLawType();

    HardeningLawFactory<NumDimT, NumSlipT> hardening_law_factory;

    auto
    phardening = hardening_law_factory.template createHardeningLaw<ArgT>(type_hardening_law);

    phardening->harden(
      slip_family,
      slip_systems,
      dt,
      rate_slip,
      state_hardening_n,
      state_hardening_np1,
      slip_resistance,
      failed);
  }

  return;
}


///
/// Update the plastic slips
///
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::updateSlip(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    minitensor::Vector<ArgT, NumSlipT> const & slip_resistance,
    minitensor::Vector<ArgT, NumSlipT> const & shear,
    minitensor::Vector<RealType, NumSlipT> const & slip_n,
    minitensor::Vector<ArgT, NumSlipT> & slip_np1,
    bool & failed)
{
  for (unsigned int ss_index(0); ss_index < slip_systems.size(); ++ ss_index)
  {
    auto const &
    slip_family = slip_families[slip_systems.at(ss_index).slip_family_index_];

    auto
    type_flow_rule = slip_family.getFlowRuleType();

    FlowRuleFactory
    flow_rule_factory;

    auto
    pflow = flow_rule_factory.template createFlowRule<ArgT>(type_flow_rule);

    ArgT
    rate_slip = pflow->computeRateSlip(
        slip_family.pflow_parameters_,
        shear[ss_index],
        slip_resistance[ss_index],
        failed);

    // return with failed immediately if slip increment is too large
    if (std::abs(rate_slip) > slip_family.pflow_parameters_->max_incr_) {
      failed = true;
      return;
    }

    slip_np1[ss_index] = slip_n[ss_index] + dt * rate_slip;
  }

  return;
}


///
/// Compute the stresses
///
template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename ArgT>
void
CP::computeStress(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    minitensor::Tensor4<ArgT, NumDimT> const & C,
    minitensor::Tensor<ArgT, NumDimT> const & F,
    minitensor::Tensor<ArgT, NumDimT> const & Fp,
    minitensor::Tensor<ArgT, NumDimT> & sigma,
    minitensor::Tensor<ArgT, NumDimT> & S,
    minitensor::Vector<ArgT, NumSlipT> & shear,
    bool & failed)
{
  minitensor::Index const
  num_dim = F.get_dimension();

  minitensor::Index const
  num_slip = shear.get_dimension();

  minitensor::Tensor<ArgT, NumDimT>
  defgrad_elastic(num_dim);

  minitensor::Tensor<ArgT, NumDimT>
  strain_elastic(num_dim);

  minitensor::Tensor<ArgT, NumDimT>
  deformation_elastic(num_dim);

  // max tolerance for Fp
  RealType
  max_tol{1.0e100};

  ArgT
  max_fp = Sacado::ScalarValue<ArgT>::eval(
         minitensor::norm_infinity(Fp));

  if (max_fp > max_tol) {
    std::cout << "Large plastic deformation gradient" << std::endl;
    std::cout << std::setprecision(4) << Fp << std::endl;
    failed = true;
    return;
  }

  ArgT
  det_fp = minitensor::det(Fp);

  // Saint Venant–Kirchhoff model
  if (det_fp == 0.0)
  {
    std::cout << "Singular plastic deformation gradient" << std::endl;
    std::cout << std::setprecision(4) << Fp << std::endl;
    failed = true;
    return;
  }

  defgrad_elastic = F * minitensor::inverse(Fp);

  ArgT
  det_fe = minitensor::det(defgrad_elastic);

  if (det_fe == 0.0) {
    std::cout << "Singular elastic deformation gradient" << std::endl;
    std::cout << std::setprecision(4) << defgrad_elastic << std::endl;
    failed = true;
    return;
  } else if (std::abs(det_fe) < std::sqrt(CP::TINY)) {
    // Downstream calculation of derivatives of 1/det_fe will fail
    failed = true;
    return;
  }

  deformation_elastic = minitensor::transpose(defgrad_elastic) * defgrad_elastic;

  strain_elastic =
    0.5 * (deformation_elastic - minitensor::identity<ArgT, NumDimT>(num_dim));

  // Not using minitensor::dotdot since C is 3x3x3x3 while strain_elastic could be 2x2
  // S = minitensor::dotdot(C, strain_elastic);
  S.fill(minitensor::Filler::ZEROS);
  for (int i = 0; i < num_dim; ++i) {
    for (int j = 0; j < num_dim; ++j) {
      for (int k = 0; k < num_dim; ++k) {
        for (int l = 0; l < num_dim; ++l) {
          S(i,j) += C(i,j,k,l) * strain_elastic(k,l);
        }
      }
    }
  }

  sigma = 1.0 / det_fe *
    defgrad_elastic * S * minitensor::transpose(defgrad_elastic);

  CP::expectFiniteTensor(
      sigma,
      "Cauchy stress in ResidualSlipNLS::computeStress()");

  // Compute resolved shear stresses
  minitensor::Tensor<ArgT,NumDimT> const
  s_trans = deformation_elastic * S;
  shear.fill(minitensor::Filler::ZEROS); 
  for (minitensor::Index s(0); s < num_slip; ++s) {
    for (int i = 0; i < num_dim; ++i) {
      for (int j = 0; j < num_dim; ++j) {
        shear[s] += slip_systems.at(s).projector_(i,j) * s_trans(i,j);
      }
    }
  }
}


//
//! Construct elasticity tensor
//
template<minitensor::Index NumDimT, typename DataT, typename ArgT>
void
CP::computeElasticityTensor(
    DataT c11,
    DataT c12,
    DataT c13,
    DataT c33,
    DataT c44,
    DataT c66,
    minitensor::Tensor4<ArgT, NumDimT> & C)
{

  C.fill(minitensor::Filler::ZEROS);

  C(0, 0, 0, 0) = c11;
  C(1, 1, 1, 1) = c11;
  C(0, 0, 1, 1) = c12;
  C(0, 1, 0, 1) = c66;
  C(0, 0, 2, 2) = c13;
  C(2, 2, 2, 2) = c33;
  C(1, 1, 2, 2) = c13;
  C(1, 2, 1, 2) = c44;
  C(0, 2, 0, 2) = c44;

  for (minitensor::Index dim_i = 0; dim_i < CP::MAX_DIM; ++dim_i) {
    for (minitensor::Index dim_j = dim_i + 1; dim_j < CP::MAX_DIM; ++dim_j) {
      C(dim_j, dim_j, dim_i, dim_i) = C(dim_i, dim_i, dim_j, dim_j);
      C(dim_j, dim_i, dim_j, dim_i) = C(dim_i, dim_j, dim_i, dim_j);
      C(dim_i, dim_j, dim_j, dim_i) = C(dim_i, dim_j, dim_i, dim_j);
      C(dim_j, dim_i, dim_i, dim_j) = C(dim_i, dim_j, dim_i, dim_j);
    }
  }
}
