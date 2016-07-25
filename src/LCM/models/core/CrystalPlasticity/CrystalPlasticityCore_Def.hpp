//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/math/special_functions/fpclassify.hpp>

///
/// Verify that constitutive update has preserved finite values
///
template<Intrepid2::Index NumDimT, typename ArgT>
void
CP::confirmTensorSanity(
    Intrepid2::Tensor<ArgT, NumDimT> const & input,
    std::string const & message)
{
  int dim = input.get_dimension();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      assert(boost::math::isfinite(Sacado::ScalarValue<ArgT>::eval(input(i, j)))==true);
      // Disabling this capability for release.
      // We will revisit this option when we can cut the time step from the constitutive model.
      /* if (!boost::math::isfinite(
          Sacado::ScalarValue<ArgT>::eval(input(i, j)))) {
        std::string msg =
            "**** Invalid data detected in CP::confirmTensorSanity(): "
                + message;
        TEUCHOS_TEST_FOR_EXCEPTION(
            !boost::math::isfinite(
                Sacado::ScalarValue<ArgT>::eval(input(i, j))),
            std::logic_error,
            msg);
      } */
    }
  }
}



///
/// Update the plastic quantities
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
CP::applySlipIncrement(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_np1,
    Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
    Intrepid2::Tensor<ArgT, NumDimT> & Lp_np1,
    Intrepid2::Tensor<ArgT, NumDimT> & Fp_np1)
{
  Intrepid2::Index const
  num_slip = slip_n.get_dimension();

  Intrepid2::Index const
  num_dim = Fp_n.get_dimension();

  // 
  // calculate plastic velocity gradient
  //
  Intrepid2::Tensor<ArgT, NumDimT>
  exp_L_dt(num_dim);

  Lp_np1.fill(Intrepid2::ZEROS);

  if(dt > 0){
    for (int s(0); s < num_slip; ++s) {
      Lp_np1 += (slip_np1[s] - slip_n[s])/dt * slip_systems[s].projector_;
    }
  }

  CP::confirmTensorSanity<NumDimT>(Lp_np1, "Lp_np1 in applySlipIncrement().");

  // update plastic deformation gradient
  // F^{p}_{n+1} = exp(L_{n+1} * delta t) F^{p}_{n}
  exp_L_dt = Intrepid2::exp(Lp_np1 * dt);
  Fp_np1 = exp_L_dt * Fp_n;

  CP::confirmTensorSanity<NumDimT>(Fp_np1, "Fp_np1 in applySlipIncrement()");
}




///
/// Evolve the hardnesses
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
CP::updateHardness(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance)
{
  for (int sf_index(0); sf_index < slip_families.size(); ++ sf_index)
  {
    auto const &
    slip_family = slip_families[sf_index];

    CP::hardeningLawFactory<NumDimT, NumSlipT, ArgT>
    f;

    CP::HardeningLawType const
    type_hardening_law = slip_family.type_hardening_law_;

    CP::HardeningLawBase<NumDimT, NumSlipT, ArgT> *
    phardening = f.createHardeningLaw(type_hardening_law);

    phardening->harden(
      slip_family,
      slip_systems,
      dt, 
      rate_slip, 
      state_hardening_n, 
      state_hardening_np1, 
      slip_resistance);
  }

  return;
}




///
/// Update the plastic slips
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
void
CP::updateSlip(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & slip_resistance,
    Intrepid2::Vector<ArgT, NumSlipT> const & shear,
    Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_np1)
{
  for (int ss_index(0); ss_index < slip_systems.size(); ++ ss_index)
  {
    auto const &
    slip_family = slip_families[slip_systems[ss_index].slip_family_index_];

    CP::flowRuleFactory<ArgT>
    f;

    CP::FlowRuleBase<ArgT> *
    pflow = f.createFlowRule(slip_family.type_flow_rule_);

    ArgT const
    rate_slip = pflow->computeRateSlip(
        slip_family.pflow_parameters_,
        shear[ss_index],
        slip_resistance[ss_index]);

    slip_np1[ss_index] = slip_n[ss_index] + dt * rate_slip;
  }

  return;
}



///
/// Compute the stresses 
///
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT, 
typename DataT>
void
CP::computeStress(
    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
    Intrepid2::Tensor4<DataT, NumDimT> const & C,
    Intrepid2::Tensor<DataT, NumDimT> const & F,
    Intrepid2::Tensor<ArgT, NumDimT> const & Fp,
    Intrepid2::Tensor<ArgT, NumDimT> & sigma,
    Intrepid2::Tensor<ArgT, NumDimT> & S,
    Intrepid2::Vector<ArgT, NumSlipT> & shear)
{
  Intrepid2::Index const
  num_dim = F.get_dimension();

  Intrepid2::Index const
  num_slip = shear.get_dimension();

  Intrepid2::Tensor<ArgT, NumDimT>
  defgrad_elastic(num_dim);

  Intrepid2::Tensor<ArgT, NumDimT>
  strain_elastic(num_dim);

  Intrepid2::Tensor<ArgT, NumDimT>
  deformation_elastic(num_dim);

  // Saint Venant–Kirchhoff model
  if (Intrepid2::det(Fp) == 0.0)
  {
    std::cout << "Singular plastic deformation gradient" << std::endl;
    std::cout << std::setprecision(4) << Fp << std::endl;
  }

  defgrad_elastic = F * Intrepid2::inverse(Fp);

  deformation_elastic = Intrepid2::transpose(defgrad_elastic) * defgrad_elastic;

  strain_elastic = 
    0.5 * (deformation_elastic - Intrepid2::identity<ArgT, NumDimT>(num_dim));

  S = Intrepid2::dotdot(C, strain_elastic);

  sigma = 1.0 / Intrepid2::det(defgrad_elastic) * 
    defgrad_elastic * S * Intrepid2::transpose(defgrad_elastic);

  CP::confirmTensorSanity<NumDimT>(
      sigma,
      "Cauchy stress in ResidualSlipNLS::computeStress()");

  // Compute resolved shear stresses
  for (int s(0); s < num_slip; ++s) {
    shear[s] = 
      Intrepid2::dotdot(slip_systems[s].projector_, deformation_elastic * S);
  }
}



//
//! Construct elasticity tensor
//
template<Intrepid2::Index NumDimT, typename DataT, typename ArgT>
void
CP::computeCubicElasticityTensor(
    DataT c11, 
    DataT c12, 
    DataT c44,
    Intrepid2::Tensor4<ArgT, NumDimT> & C)
{

  Intrepid2::Index const
  num_dim = C.get_dimension();

  C.fill(Intrepid2::ZEROS);

  for (Intrepid2::Index dim_i = 0; dim_i < num_dim; ++dim_i) {
    C(dim_i, dim_i, dim_i, dim_i) = c11;
    for (Intrepid2::Index dim_j = dim_i + 1; dim_j < num_dim; ++dim_j) {
      C(dim_i, dim_i, dim_j, dim_j) = c12;
      C(dim_j, dim_j, dim_i, dim_i) = C(dim_i, dim_i, dim_j, dim_j);
      C(dim_i, dim_j, dim_i, dim_j) = c44;
      C(dim_j, dim_i, dim_j, dim_i) = C(dim_i, dim_j, dim_i, dim_j);
      C(dim_i, dim_j, dim_j, dim_i) = C(dim_i, dim_j, dim_i, dim_j);
      C(dim_j, dim_i, dim_i, dim_j) = C(dim_i, dim_j, dim_i, dim_j);
    }
  }
}
