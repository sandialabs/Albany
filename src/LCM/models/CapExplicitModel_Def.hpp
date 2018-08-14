//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <MiniTensor.h>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>
#include <typeinfo>

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
CapExplicitModel<EvalT, Traits>::CapExplicitModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      A(p->get<RealType>("A")),
      B(p->get<RealType>("B")),
      C(p->get<RealType>("C")),
      theta(p->get<RealType>("theta")),
      R(p->get<RealType>("R")),
      kappa0(p->get<RealType>("kappa0")),
      W(p->get<RealType>("W")),
      D1(p->get<RealType>("D1")),
      D2(p->get<RealType>("D2")),
      calpha(p->get<RealType>("calpha")),
      psi(p->get<RealType>("psi")),
      N(p->get<RealType>("N")),
      L(p->get<RealType>("L")),
      phi(p->get<RealType>("phi")),
      Q(p->get<RealType>("Q"))
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Strain", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // retrieve appropriate field name strings
  std::string cauchy_string           = (*field_name_map_)["Cauchy_Stress"];
  std::string strain_string           = (*field_name_map_)["Strain"];
  std::string backStress_string       = (*field_name_map_)["Back_Stress"];
  std::string capParameter_string     = (*field_name_map_)["Cap_Parameter"];
  std::string eqps_string             = (*field_name_map_)["eqps"];
  std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(
      std::make_pair(backStress_string, dl->qp_tensor));
  this->eval_field_map_.insert(
      std::make_pair(capParameter_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair(volPlasticStrain_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair("Material Tangent", dl->qp_tensor4));

  // define the state variables
  //
  // strain
  this->num_state_variables_++;
  this->state_var_names_.push_back(strain_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // backStress
  this->num_state_variables_++;
  this->state_var_names_.push_back(backStress_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // capParameter
  this->num_state_variables_++;
  this->state_var_names_.push_back(capParameter_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(kappa0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // volPlasticStrain
  this->num_state_variables_++;
  this->state_var_names_.push_back(volPlasticStrain_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  // initialize tensor
  //
  I             = minitensor::eye<ScalarT>(num_dims_);
  id1           = minitensor::identity_1<ScalarT>(num_dims_);
  id2           = minitensor::identity_2<ScalarT>(num_dims_);
  id3           = minitensor::identity_3<ScalarT>(num_dims_);
  Celastic      = minitensor::Tensor4<ScalarT>(num_dims_);
  compliance    = minitensor::Tensor4<ScalarT>(num_dims_);
  depsilon      = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaN        = minitensor::Tensor<ScalarT>(num_dims_);
  strainN       = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaVal      = minitensor::Tensor<ScalarT>(num_dims_);
  alphaVal      = minitensor::Tensor<ScalarT>(num_dims_);
  deps_plastic  = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaTr       = minitensor::Tensor<ScalarT>(num_dims_);
  alphaTr       = minitensor::Tensor<ScalarT>(num_dims_);
  dfdsigma      = minitensor::Tensor<ScalarT>(num_dims_);
  dgdsigma      = minitensor::Tensor<ScalarT>(num_dims_);
  dfdalpha      = minitensor::Tensor<ScalarT>(num_dims_);
  halpha        = minitensor::Tensor<ScalarT>(num_dims_);
  dfdotCe       = minitensor::Tensor<ScalarT>(num_dims_);
  sigmaK        = minitensor::Tensor<ScalarT>(num_dims_);
  alphaK        = minitensor::Tensor<ScalarT>(num_dims_);
  dsigma        = minitensor::Tensor<ScalarT>(num_dims_);
  dev_plastic   = minitensor::Tensor<ScalarT>(num_dims_);
  xi            = minitensor::Tensor<ScalarT>(num_dims_);
  sN            = minitensor::Tensor<ScalarT>(num_dims_);
  s             = minitensor::Tensor<ScalarT>(num_dims_);
  strainCurrent = minitensor::Tensor<ScalarT>(num_dims_);
  dJ3dsigma     = minitensor::Tensor<ScalarT>(num_dims_);
  eps_dev       = minitensor::Tensor<ScalarT>(num_dims_);
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
CapExplicitModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // extract dependent MDFields
  auto strain          = *dep_fields["Strain"];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];

  // retrieve appropriate field name strings
  std::string cauchy_string           = (*field_name_map_)["Cauchy_Stress"];
  std::string strain_string           = (*field_name_map_)["Strain"];
  std::string backStress_string       = (*field_name_map_)["Back_Stress"];
  std::string capParameter_string     = (*field_name_map_)["Cap_Parameter"];
  std::string eqps_string             = (*field_name_map_)["eqps"];
  std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];

  // extract evaluated MDFields
  auto stress           = *eval_fields[cauchy_string];
  auto backStress       = *eval_fields[backStress_string];
  auto capParameter     = *eval_fields[capParameter_string];
  auto eqps             = *eval_fields[eqps_string];
  auto volPlasticStrain = *eval_fields[volPlasticStrain_string];
  auto tangent          = *eval_fields["Material Tangent"];

  // get State Variables
  Albany::MDArray strainold = (*workset.stateArrayPtr)[strain_string + "_old"];
  Albany::MDArray stressold = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  Albany::MDArray backStressold =
      (*workset.stateArrayPtr)[backStress_string + "_old"];
  Albany::MDArray capParameterold =
      (*workset.stateArrayPtr)[capParameter_string + "_old"];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray volPlasticStrainold =
      (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];

  ScalarT lame, mu, bulkModulus;
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < num_pts_; ++qp) {
      // local parameters
      lame = elastic_modulus(cell, qp) * poissons_ratio(cell, qp) /
             (1.0 + poissons_ratio(cell, qp)) /
             (1.0 - 2.0 * poissons_ratio(cell, qp));
      mu = elastic_modulus(cell, qp) / 2.0 / (1.0 + poissons_ratio(cell, qp));
      bulkModulus = lame + (2. / 3.) * mu;

      // elastic matrix
      Celastic = lame * id3 + mu * (id1 + id2);

      // elastic compliance tangent matrix
      compliance = (1. / bulkModulus / 9.) * id3 +
                   (1. / mu / 2.) * (0.5 * (id1 + id2) - (1. / 3.) * id3);

      // trial state
      minitensor::Tensor<ScalarT> depsilon(3);
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);
          strainN(i, j)  = strainold(cell, qp, i, j);
          sigmaN(i, j)   = stressold(cell, qp, i, j);
          alphaVal(i, j) = backStressold(cell, qp, i, j);
        }
      }

      sigmaVal         = sigmaN + minitensor::dotdot(Celastic, depsilon);
      ScalarT kappaVal = capParameterold(cell, qp);

      // initialize friction and dilatancy
      // (which will be updated only if plasticity occurs)
      // friction(cell, qp) = 0.0;
      // dilatancy(cell, qp) = 0.0;
      // hardening_modulus(cell, qp) = 0.0;

      // define generalized plastic hardening modulus H
      // ScalarT H(0.0), Htan(0.0);

      // define plastic strain increment, its two invariants: dev, and vol
      ScalarT deqps(0.0), devolps(0.0);

      // define a temporary tensor to store trial stress tensors
      sigmaTr = sigmaVal;
      // define a temporary tensor to store previous back stress
      alphaTr = alphaVal;

      // check yielding
      ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);

      // plastic correction
      ScalarT dgamma = 0.0;
      if (f > 0.0) {
        dfdsigma = compute_dfdsigma(sigmaN, alphaVal, kappaVal);

        dgdsigma = compute_dgdsigma(sigmaN, alphaVal, kappaVal);

        dfdalpha = -dfdsigma;

        ScalarT dfdkappa = compute_dfdkappa(sigmaN, alphaVal, kappaVal);

        ScalarT J2_alpha = 0.5 * minitensor::dotdot(alphaVal, alphaVal);

        halpha = compute_halpha(dgdsigma, J2_alpha);

        ScalarT I1_dgdsigma = minitensor::trace(dgdsigma);

        ScalarT dedkappa = compute_dedkappa(kappaVal);

        ScalarT hkappa;
        if (dedkappa != 0.0)
          hkappa = I1_dgdsigma / dedkappa;
        else
          hkappa = 0.0;

        ScalarT kai(0.0);
        kai = minitensor::dotdot(
                  dfdsigma, minitensor::dotdot(Celastic, dgdsigma)) -
              minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

        // H = -minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

        dfdotCe = minitensor::dotdot(dfdsigma, Celastic);

        if (kai != 0.0)
          dgamma = minitensor::dotdot(dfdotCe, depsilon) / kai;
        else
          dgamma = 0.0;

        // update
        sigmaVal -= dgamma * minitensor::dotdot(Celastic, dgdsigma);

        alphaVal += dgamma * halpha;

        // restrictions on kappa, only allow monotonic decreasing (cap
        // hardening)
        ScalarT dkappa = dgamma * hkappa;
        if (dkappa > 0) {
          dkappa = 0;
          // H = -minitensor::dotdot(dfdalpha, halpha);
        }

        kappaVal += dkappa;

        // stress correction algorithm to avoid drifting from yield surface
        bool     condition     = false;
        int      iteration     = 0;
        int      max_iteration = 20;
        RealType tolerance     = 1.0e-10;
        while (condition == false) {
          f = compute_f(sigmaVal, alphaVal, kappaVal);

          dfdsigma = compute_dfdsigma(sigmaVal, alphaVal, kappaVal);

          dgdsigma = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);

          dfdalpha = -dfdsigma;

          ScalarT dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);

          J2_alpha = 0.5 * minitensor::dotdot(alphaVal, alphaVal);

          halpha = compute_halpha(dgdsigma, J2_alpha);

          I1_dgdsigma = minitensor::trace(dgdsigma);

          dedkappa = compute_dedkappa(kappaVal);

          if (dedkappa != 0)
            hkappa = I1_dgdsigma / dedkappa;
          else
            hkappa = 0;

          // generalized plastic hardening modulus
          // H = -minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          kai = minitensor::dotdot(
              dfdsigma, minitensor::dotdot(Celastic, dgdsigma));
          kai = kai - minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          if (std::abs(f) < tolerance) break;
          if (iteration > max_iteration) {
            // output for debug
            // std::cout << "no stress correction after iteration = "
            //<< iteration << " yield function abs(f) = " << abs(f)
            //<< std::endl;
            break;
          }

          ScalarT delta_gamma;
          if (kai != 0)
            delta_gamma = f / kai;
          else
            delta_gamma = 0;

          // restrictions on kappa, only allow monotonic decreasing
          dkappa = delta_gamma * hkappa;
          if (dkappa > 0.0) {
            dkappa = 0.0;
            // H = -minitensor::dotdot(dfdalpha, halpha);
          }

          // update
          sigmaK =
              sigmaVal - delta_gamma * minitensor::dotdot(Celastic, dgdsigma);
          alphaK         = alphaVal + delta_gamma * halpha;
          ScalarT kappaK = kappaVal + dkappa;

          ScalarT fK = compute_f(sigmaK, alphaK, kappaK);

          if (std::abs(fK) > std::abs(f)) {
            // if the corrected stress is further away from yield surface,
            // then use normal correction
            ScalarT dfdotdf = minitensor::dotdot(dfdsigma, dfdsigma);
            if (dfdotdf != 0)
              delta_gamma = f / dfdotdf;
            else
              delta_gamma = 0.0;

            sigmaK = sigmaVal - delta_gamma * dfdsigma;
            alphaK = alphaVal;
            kappaK = kappaVal;

            // H = 0.0;
          }

          sigmaVal = sigmaK;
          alphaVal = alphaK;
          kappaVal = kappaK;

          iteration++;

        }  // end of stress correction

        // compute plastic strain increment
        // deps_plastic = compliance ( sigma_tr - sigma_(n+1));
        dsigma       = sigmaTr - sigmaVal;
        deps_plastic = minitensor::dotdot(compliance, dsigma);

        // compute its two invariants
        // devolps (volumetric) and deqps (deviatoric)
        devolps     = minitensor::trace(deps_plastic);
        dev_plastic = deps_plastic - (1. / 3.) * devolps * I;
        // use altenative definition, differ by constants
        deqps = std::sqrt(2) * minitensor::norm(dev_plastic);

        // dilatancy
        // if (deqps != 0)
        //  dilatancy(cell, qp) = devolps / deqps;
        // else
        //  dilatancy(cell, qp) = 0.0;

        // previous p and tau
        // ScalarT pN(0.0), tauN(0.0);
        // xi = sigmaN - alphaTr;
        // pN = minitensor::trace(xi);
        // pN = pN / 3.;
        // sN = xi - pN * I;
        // tauN = sqrt(1. / 2.) * minitensor::norm(sN);

        // current p, and tau
        // ScalarT p(0.0), tau(0.0);
        // xi = sigmaVal - alphaVal;
        // p = minitensor::trace(xi);
        // p = p / 3.;
        // s = xi - p * I;
        // tau = sqrt(1. / 2.) * minitensor::norm(s);

        // difference
        // ScalarT dtau = tau - tauN;
        // ScalarT dp = p - pN;

        // friction coefficient by finite difference
        // if (dp != 0)
        //  friction(cell, qp) = dtau / dp;
        // else
        //  friction(cell, qp) = 0.0;

        // previous gamma(gamma)
        // ScalarT evol3 = minitensor::trace(strainN);
        // evol3 = evol3 / 3.;
        // eps_dev = strainN - evol3 * I;
        // ScalarT gammaN = sqrt(2.) * minitensor::norm(eps_dev);

        // current gamma(gamma)
        // strainCurrent = strainN + depsilon;
        // evol3 = minitensor::trace(strainCurrent);
        // evol3 = evol3 / 3.;
        // eps_dev = strainCurrent - evol3 * I;
        // ScalarT gamma = sqrt(2.) * minitensor::norm(eps_dev);

        // difference
        // ScalarT dGamma = gamma - gammaN;
        // tagent hardening modulus
        // if (dGamma != 0)
        //  Htan = dtau / dGamma;

        // if (std::abs(1. - Htan / mu) > 1.0e-10)
        //  hardening_modulus(cell, qp) = Htan / (1. - Htan / mu);
        // else
        //  hardening_modulus(cell, qp) = 0.0;

      }  // end of plastic correction

      // update
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, qp, i, j)     = sigmaVal(i, j);
          backStress(cell, qp, i, j) = alphaVal(i, j);
        }
      }

      capParameter(cell, qp)     = kappaVal;
      eqps(cell, qp)             = eqpsold(cell, qp) + deqps;
      volPlasticStrain(cell, qp) = volPlasticStrainold(cell, qp) + devolps;

    }  // loop over qps

  }  // loop over cell

}  // end of evaluateFields
//------------------------------------------------------------------------------
// all local functions
template <typename EvalT, typename Traits>
typename CapExplicitModel<EvalT, Traits>::ScalarT
// typename EvalT::ScalarT
CapExplicitModel<EvalT, Traits>::compute_f(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT&                     kappa)
{
  xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

  s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  ScalarT Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma =
        0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
               (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

  ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

  ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

  ScalarT X = kappa - R * Ff_kappa;

  ScalarT Fc = 1.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitModel<EvalT, Traits>::ScalarT>
CapExplicitModel<EvalT, Traits>::compute_dfdsigma(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT&                     kappa)
{
  xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

  s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  // dI1dsigma = I;
  // dJ2dsigma = s;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * I(i, j) / 3;

  ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

  ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

  ScalarT X = kappa - R * Ff_kappa;

  ScalarT Fc = 1.0;

  if ((kappa - I1) > 0)
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma =
        0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
               (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

  // derivatives
  ScalarT dFfdI1 = -(B * C * std::exp(B * I1) + theta);

  ScalarT dFcdI1 = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

  ScalarT dGammadJ2 = 0.0;
  if (J2 != 0)
    dGammadJ2 =
        9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);

  ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

  ScalarT dGammadJ3 = 0.0;
  if (J2 != 0)
    dGammadJ3 =
        -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);

  ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

  dfdsigma = dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;

  return dfdsigma;
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitModel<EvalT, Traits>::ScalarT>
CapExplicitModel<EvalT, Traits>::compute_dgdsigma(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT&                     kappa)
{
  xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3;

  s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  // dJ2dsigma = s;
  // dJ3dsigma(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * I(i, j) / 3;

  ScalarT Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

  ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

  ScalarT X = kappa - Q * Ff_kappa;

  ScalarT Fc = 1.0;

  if ((kappa - I1) > 0)
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT Gamma = 1.0;
  if (psi != 0 && J2 != 0)
    Gamma =
        0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5) +
               (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

  // derivatives
  ScalarT dFfdI1 = -(L * C * std::exp(L * I1) + phi);

  ScalarT dFcdI1 = 0.0;
  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

  ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

  ScalarT dGammadJ2 = 0.0;
  if (J2 != 0)
    dGammadJ2 =
        9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);

  ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

  ScalarT dGammadJ3 = 0.0;
  if (J2 != 0)
    dGammadJ3 =
        -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);

  ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

  dgdsigma = dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;

  return dgdsigma;
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename CapExplicitModel<EvalT, Traits>::ScalarT
CapExplicitModel<EvalT, Traits>::compute_dfdkappa(
    minitensor::Tensor<ScalarT>& sigma,
    minitensor::Tensor<ScalarT>& alpha,
    ScalarT&                     kappa)
{
  ScalarT dfdkappa;

  xi = sigma - alpha;

  ScalarT I1 = minitensor::trace(xi), p = I1 / 3.0;

  s = xi - p * minitensor::identity<ScalarT>(3);

  ScalarT J2 = 0.5 * minitensor::dotdot(s, s);

  ScalarT J3 = minitensor::det(s);

  ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

  ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

  ScalarT X = kappa - R * Ff_kappa;

  ScalarT dFcdkappa = 0.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0)) {
    dFcdkappa = 2 * (I1 - kappa) *
                ((X - kappa) +
                 R * (I1 - kappa) * (theta + B * C * std::exp(B * kappa))) /
                (X - kappa) / (X - kappa) / (X - kappa);
  }

  dfdkappa = -dFcdkappa * (Ff_I1 - N) * (Ff_I1 - N);

  return dfdkappa;
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename CapExplicitModel<EvalT, Traits>::ScalarT
CapExplicitModel<EvalT, Traits>::compute_Galpha(ScalarT& J2_alpha)
{
  if (N != 0)
    return 1.0 - std::pow(J2_alpha, 0.5) / N;
  else
    return 0.0;
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapExplicitModel<EvalT, Traits>::ScalarT>
CapExplicitModel<EvalT, Traits>::compute_halpha(
    minitensor::Tensor<ScalarT>& dgdsigma,
    ScalarT&                     J2_alpha)
{
  ScalarT Galpha = compute_Galpha(J2_alpha);

  ScalarT I1 = minitensor::trace(dgdsigma), p = I1 / 3;

  s = dgdsigma - p * I;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { halpha(i, j) = calpha * Galpha * s(i, j); }
  }

  return halpha;
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename CapExplicitModel<EvalT, Traits>::ScalarT
CapExplicitModel<EvalT, Traits>::compute_dedkappa(ScalarT& kappa)
{
  ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;

  ScalarT X0 = kappa0 - Q * Ff_kappa0;

  ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

  ScalarT X = kappa - Q * Ff_kappa;

  ScalarT dedX =
      (D1 - 2 * D2 * (X - X0)) * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

  ScalarT dXdkappa = 1 + Q * C * L * std::exp(L * kappa) + Q * phi;

  return dedX * dXdkappa;
}
//------------------------------------------------------------------------------
}  // namespace LCM
