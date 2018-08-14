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

#include "LocalNonlinearSolver.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
CapImplicitModel<EvalT, Traits>::CapImplicitModel(
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

  // optional material tangent computation
  std::string tangent_string = (*field_name_map_)["Material Tangent"];

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

  if (compute_tangent_) {
    this->eval_field_map_.insert(
        std::make_pair(tangent_string, dl->qp_tensor4));
  }

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
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
CapImplicitModel<EvalT, Traits>::computeState(
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
  std::string tangent_string          = (*field_name_map_)["Material Tangent"];

  // extract evaluated MDFields
  auto stress           = *eval_fields[cauchy_string];
  auto backStress       = *eval_fields[backStress_string];
  auto capParameter     = *eval_fields[capParameter_string];
  auto eqps             = *eval_fields[eqps_string];
  auto volPlasticStrain = *eval_fields[volPlasticStrain_string];
  PHX::MDField<ScalarT> tangent;
  if (compute_tangent_) tangent = *eval_fields[tangent_string];

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

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < num_pts_; ++qp) {
      // local parameters
      ScalarT lame = elastic_modulus(cell, qp) * poissons_ratio(cell, qp) /
                     (1.0 + poissons_ratio(cell, qp)) /
                     (1.0 - 2.0 * poissons_ratio(cell, qp));
      ScalarT mu =
          elastic_modulus(cell, qp) / 2.0 / (1.0 + poissons_ratio(cell, qp));
      ScalarT bulkModulus = lame + (2. / 3.) * mu;

      // elastic matrix
      minitensor::Tensor4<ScalarT> Celastic =
          lame * minitensor::identity_3<ScalarT>(3) +
          mu * (minitensor::identity_1<ScalarT>(3) +
                minitensor::identity_2<ScalarT>(3));

      // elastic compliance tangent matrix
      minitensor::Tensor4<ScalarT> compliance =
          (1. / bulkModulus / 9.) * minitensor::identity_3<ScalarT>(3) +
          (1. / mu / 2.) * (0.5 * (minitensor::identity_1<ScalarT>(3) +
                                   minitensor::identity_2<ScalarT>(3)) -
                            (1. / 3.) * minitensor::identity_3<ScalarT>(3));

      // previous state
      minitensor::Tensor<ScalarT> sigmaN(3, minitensor::Filler::ZEROS),
          alphaN(3, minitensor::Filler::ZEROS),
          strainN(3, minitensor::Filler::ZEROS);

      // incremental strain tensor
      minitensor::Tensor<ScalarT> depsilon(3);
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);
          strainN(i, j)  = strainold(cell, qp, i, j);
        }
      }

      // trial state
      minitensor::Tensor<ScalarT> sigmaVal =
          minitensor::dotdot(Celastic, depsilon);
      minitensor::Tensor<ScalarT> alphaVal(3, minitensor::Filler::ZEROS);

      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          sigmaVal(i, j) = sigmaVal(i, j) + stressold(cell, qp, i, j);
          alphaVal(i, j) = backStressold(cell, qp, i, j);
          sigmaN(i, j)   = stressold(cell, qp, i, j);
          alphaN(i, j)   = backStressold(cell, qp, i, j);
        }
      }

      ScalarT kappaVal  = capParameterold(cell, qp);
      ScalarT dgammaVal = 0.0;

      // used in defining generalized hardening modulus
      ScalarT Htan(0.0);

      // define plastic strain increment, its two invariants: dev, and vol
      minitensor::Tensor<ScalarT> deps_plastic(3, minitensor::Filler::ZEROS);
      ScalarT                     deqps(0.0), devolps(0.0);

      // define temporary trial stress, used in computing plastic strain
      minitensor::Tensor<ScalarT> sigmaTr = sigmaVal;

      std::vector<ScalarT> XXVal(13);

      // check yielding
      ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);
      XXVal     = initialize(sigmaVal, alphaVal, kappaVal, dgammaVal);

      // local Newton loop
      if (f > 1.e-11) {  // plastic yielding

        ScalarT normR, normR0, conv;
        bool    kappa_flag = false;
        bool    converged  = false;
        int     iter       = 0;

        std::vector<ScalarT>                R(13);
        std::vector<ScalarT>                dRdX(13 * 13);
        LocalNonlinearSolver<EvalT, Traits> solver;

        while (!converged) {
          // assemble residual vector and local Jacobian
          compute_ResidJacobian(
              XXVal,
              R,
              dRdX,
              sigmaVal,
              alphaVal,
              kappaVal,
              Celastic,
              kappa_flag);

          normR = 0.0;
          for (int i = 0; i < 13; i++) normR += R[i] * R[i];

          normR = std::sqrt(normR);

          if (iter == 0) normR0 = normR;
          if (normR0 != 0)
            conv = normR / normR0;
          else
            conv = normR0;

          if (conv < 1.e-11 || normR < 1.e-11) break;

          if (iter > 20) break;

          // TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
          // std::endl << "Error in return mapping, iter = "
          //<< iter << "\nres = " << normR << "\nrelres = " << conv <<
          //std::endl;

          std::vector<ScalarT> XXValK = XXVal;
          solver.solve(dRdX, XXValK, R);

          // put restrictions on kappa: only allows monotonic decreasing (cap
          // hardening)
          if (XXValK[11] > XXVal[11]) {
            kappa_flag = true;
          } else {
            XXVal      = XXValK;
            kappa_flag = false;
          }

          // debugging
          // XXVal = XXValK;

          iter++;
        }  // end local NR

        // compute sensitivity information, and pack back to X.
        solver.computeFadInfo(dRdX, XXVal, R);

      }  // end of plasticity

      // update
      sigmaVal(0, 0) = XXVal[0];
      sigmaVal(0, 1) = XXVal[5];
      sigmaVal(0, 2) = XXVal[4];
      sigmaVal(1, 0) = XXVal[5];
      sigmaVal(1, 1) = XXVal[1];
      sigmaVal(1, 2) = XXVal[3];
      sigmaVal(2, 0) = XXVal[4];
      sigmaVal(2, 1) = XXVal[3];
      sigmaVal(2, 2) = XXVal[2];

      alphaVal(0, 0) = XXVal[6];
      alphaVal(0, 1) = XXVal[10];
      alphaVal(0, 2) = XXVal[9];
      alphaVal(1, 0) = XXVal[10];
      alphaVal(1, 1) = XXVal[7];
      alphaVal(1, 2) = XXVal[8];
      alphaVal(2, 0) = XXVal[9];
      alphaVal(2, 1) = XXVal[8];
      alphaVal(2, 2) = -XXVal[6] - XXVal[7];

      kappaVal = XXVal[11];

      dgammaVal = XXVal[12];

      // compute plastic strain increment deps_plastic = compliance ( sigma_tr -
      // sigma_(n+1));
      minitensor::Tensor<ScalarT> dsigma = sigmaTr - sigmaVal;
      deps_plastic = minitensor::dotdot(compliance, dsigma);

      // compute its two invariants: devolps (volumetric) and deqps (deviatoric)
      devolps = minitensor::trace(deps_plastic);
      minitensor::Tensor<ScalarT> dev_plastic =
          deps_plastic -
          (1.0 / 3.0) * devolps * minitensor::identity<ScalarT>(3);
      // deqps = std::sqrt(2./3.) * minitensor::norm(dev_plastic);
      // use altenative definition, just differ by constants
      deqps = std::sqrt(2) * minitensor::norm(dev_plastic);

      // stress and back stress
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, qp, i, j)     = sigmaVal(i, j);
          backStress(cell, qp, i, j) = alphaVal(i, j);
        }
      }

      capParameter(cell, qp)     = kappaVal;
      eqps(cell, qp)             = eqpsold(cell, qp) + deqps;
      volPlasticStrain(cell, qp) = volPlasticStrainold(cell, qp) + devolps;

      if (compute_tangent_) {
        minitensor::Tensor4<ScalarT> Cep =
            compute_Cep(Celastic, sigmaVal, alphaVal, kappaVal, dgammaVal);

        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            for (int k(0); k < num_dims_; ++k) {
              for (int l(0); l < num_dims_; ++l) {
                tangent(cell, qp, i, j, k, l) = Cep(i, j, k, l);
              }
            }
          }
        }
      }

    }  // loop over qps

  }  // loop over cell

}  // end of evaluateFields

//**************************** all local functions *****************************

//------------------------------ yield function ------------------------------//
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitModel<EvalT, Traits>::compute_f(
    minitensor::Tensor<T>& sigma,
    minitensor::Tensor<T>& alpha,
    T&                     kappa)
{
  minitensor::Tensor<T> xi = sigma - alpha;

  T I1 = minitensor::trace(xi), p = I1 / 3.;

  minitensor::Tensor<T> s = xi - p * minitensor::identity<T>(3);

  T J2 = 0.5 * minitensor::dotdot(s, s);

  T J3 = minitensor::det(s);

  T Gamma = 1.0;

  if (psi != 0 && J2 != 0)
    Gamma =
        0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5) +
               (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);

  T Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

  T Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

  T X = kappa - R * Ff_kappa;

  T Fc = 1.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

//------------------------ unknow variable value list ------------------------//
template <typename EvalT, typename Traits>
std::vector<typename CapImplicitModel<EvalT, Traits>::ScalarT>
// std::vector<typename EvalT::ScalarT>
CapImplicitModel<EvalT, Traits>::initialize(
    minitensor::Tensor<ScalarT>& sigmaVal,
    minitensor::Tensor<ScalarT>& alphaVal,
    ScalarT&                     kappaVal,
    ScalarT&                     dgammaVal)
{
  std::vector<ScalarT> XX(13);

  XX[0]  = sigmaVal(0, 0);
  XX[1]  = sigmaVal(1, 1);
  XX[2]  = sigmaVal(2, 2);
  XX[3]  = sigmaVal(1, 2);
  XX[4]  = sigmaVal(0, 2);
  XX[5]  = sigmaVal(0, 1);
  XX[6]  = alphaVal(0, 0);
  XX[7]  = alphaVal(1, 1);
  XX[8]  = alphaVal(1, 2);
  XX[9]  = alphaVal(0, 2);
  XX[10] = alphaVal(0, 1);
  XX[11] = kappaVal;
  XX[12] = dgammaVal;

  return XX;
}

//----------------------- local iteration jacobian ---------------------------//
template <typename EvalT, typename Traits>
void
CapImplicitModel<EvalT, Traits>::compute_ResidJacobian(
    std::vector<ScalarT> const&         XXVal,
    std::vector<ScalarT>&               R,
    std::vector<ScalarT>&               dRdX,
    const minitensor::Tensor<ScalarT>&  sigmaVal,
    const minitensor::Tensor<ScalarT>&  alphaVal,
    const ScalarT&                      kappaVal,
    minitensor::Tensor4<ScalarT> const& Celastic,
    bool                                kappa_flag)
{
  std::vector<DFadType> Rfad(13);
  std::vector<DFadType> XX(13);
  std::vector<ScalarT>  XXtmp(13);

  // initialize DFadType local unknown vector Xfad
  // Note that since Xfad is a temporary variable that gets changed within local
  // iterations when we initialize Xfad, we only pass in the values of X, NOT
  // the system sensitivity information
  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XXVal[i]);
    XX[i]    = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XX[0];
  sigma(0, 1) = XX[5];
  sigma(0, 2) = XX[4];
  sigma(1, 0) = XX[5];
  sigma(1, 1) = XX[1];
  sigma(1, 2) = XX[3];
  sigma(2, 0) = XX[4];
  sigma(2, 1) = XX[3];
  sigma(2, 2) = XX[2];

  alpha(0, 0) = XX[6];
  alpha(0, 1) = XX[10];
  alpha(0, 2) = XX[9];
  alpha(1, 0) = XX[10];
  alpha(1, 1) = XX[7];
  alpha(1, 2) = XX[8];
  alpha(2, 0) = XX[9];
  alpha(2, 1) = XX[8];
  alpha(2, 2) = -XX[6] - XX[7];

  DFadType kappa = XX[11];

  DFadType dgamma = XX[12];

  DFadType f = compute_f(sigma, alpha, kappa);

  minitensor::Tensor<DFadType> dgdsigma = compute_dgdsigma(XX);

  DFadType J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);

  minitensor::Tensor<DFadType> halpha = compute_halpha(dgdsigma, J2_alpha);

  DFadType I1_dgdsigma = minitensor::trace(dgdsigma);

  DFadType dedkappa = compute_dedkappa(kappa);

  DFadType hkappa = compute_hkappa(I1_dgdsigma, dedkappa);

  DFadType t;

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(0, 0, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[0] = dgamma * t + sigma(0, 0) - sigmaVal(0, 0);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(1, 1, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[1] = dgamma * t + sigma(1, 1) - sigmaVal(1, 1);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(2, 2, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[2] = dgamma * t + sigma(2, 2) - sigmaVal(2, 2);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(1, 2, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[3] = dgamma * t + sigma(1, 2) - sigmaVal(1, 2);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(0, 2, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[4] = dgamma * t + sigma(0, 2) - sigmaVal(0, 2);

  t = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      t = t + Celastic(0, 1, i, j) * dgdsigma(i, j);
    }
  }
  Rfad[5] = dgamma * t + sigma(0, 1) - sigmaVal(0, 1);

  Rfad[6] = dgamma * halpha(0, 0) - alpha(0, 0) + alphaVal(0, 0);

  Rfad[7] = dgamma * halpha(1, 1) - alpha(1, 1) + alphaVal(1, 1);

  Rfad[8] = dgamma * halpha(1, 2) - alpha(1, 2) + alphaVal(1, 2);

  Rfad[9] = dgamma * halpha(0, 2) - alpha(0, 2) + alphaVal(0, 2);

  Rfad[10] = dgamma * halpha(0, 1) - alpha(0, 1) + alphaVal(0, 1);

  if (kappa_flag == false)
    Rfad[11] = dgamma * hkappa - kappa + kappaVal;
  else
    Rfad[11] = 0;

  Rfad[12] = f;

  // get ScalarT Residual
  for (int i = 0; i < 13; i++) R[i] = Rfad[i].val();

  // std::cout << "in assemble_Resid, R= " << R[0] << " " << R[1] << " " << R[2]
  // << " " << R[3]<< std::endl;

  // get Jacobian
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 13; j++) dRdX[i + 13 * j] = Rfad[i].dx(j);

  if (kappa_flag == true) {
    for (int j = 0; j < 13; j++) dRdX[11 + 13 * j] = 0.0;

    dRdX[11 + 13 * 11] = 1.0;
  }
}

//---------------------------- plastic potential -----------------------------//
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitModel<EvalT, Traits>::compute_g(
    minitensor::Tensor<T>& sigma,
    minitensor::Tensor<T>& alpha,
    T&                     kappa)
{
  minitensor::Tensor<T> xi = sigma - alpha;

  T I1 = minitensor::trace(xi);

  T p = I1 / 3.;

  minitensor::Tensor<T> s = xi - p * minitensor::identity<T>(3);

  T J2 = 0.5 * minitensor::dotdot(s, s);

  T J3 = minitensor::det(s);

  T Gamma = 1.0;

  if (psi != 0 && J2 != 0)
    Gamma =
        0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5) +
               (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);

  T Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

  T Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

  T X = kappa - Q * Ff_kappa;

  T Fc = 1.0;

  if ((kappa - I1) > 0 && ((X - kappa) != 0))
    Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

  return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

//----------------------------- derivative -----------------------------------//
template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitModel<EvalT, Traits>::ScalarT>
// minitensor::Tensor<typename EvalT::DFadType>
CapImplicitModel<EvalT, Traits>::compute_dfdsigma(
    std::vector<ScalarT> const& XX)
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XXFad[0];
  sigma(0, 1) = XXFad[5];
  sigma(0, 2) = XXFad[4];
  sigma(1, 0) = XXFad[5];
  sigma(1, 1) = XXFad[1];
  sigma(1, 2) = XXFad[3];
  sigma(2, 0) = XXFad[4];
  sigma(2, 1) = XXFad[3];
  sigma(2, 2) = XXFad[2];

  // NOTE: DFadType alpha and kappa may not be necessary
  // since we only compute dfdsigma while keeping alpha and kappa constant
  alpha(0, 0) = XXFad[6];
  alpha(0, 1) = XXFad[10];
  alpha(0, 2) = XXFad[9];
  alpha(1, 0) = XXFad[10];
  alpha(1, 1) = XXFad[7];
  alpha(1, 2) = XXFad[8];
  alpha(2, 0) = XXFad[9];
  alpha(2, 1) = XXFad[8];
  alpha(2, 2) = -XXFad[6] - XXFad[7];

  DFadType kappa = XXFad[11];

  DFadType f = compute_f(sigma, alpha, kappa);

  minitensor::Tensor<ScalarT> dfdsigma(3);

  dfdsigma(0, 0) = f.dx(0);
  dfdsigma(0, 1) = f.dx(5);
  dfdsigma(0, 2) = f.dx(4);
  dfdsigma(1, 0) = f.dx(5);
  dfdsigma(1, 1) = f.dx(1);
  dfdsigma(1, 2) = f.dx(3);
  dfdsigma(2, 0) = f.dx(4);
  dfdsigma(2, 1) = f.dx(3);
  dfdsigma(2, 2) = f.dx(2);

  return dfdsigma;
}

template <typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::ScalarT
// minitensor::Tensor<typename EvalT::ScalarT>
CapImplicitModel<EvalT, Traits>::compute_dfdkappa(
    std::vector<ScalarT> const& XX)
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  // NOTE: DFadType sigma and alpha may not be necessary
  // since we only compute dfdkappa while keeping sigma and alpha constant
  sigma(0, 0) = XXFad[0];
  sigma(0, 1) = XXFad[5];
  sigma(0, 2) = XXFad[4];
  sigma(1, 0) = XXFad[5];
  sigma(1, 1) = XXFad[1];
  sigma(1, 2) = XXFad[3];
  sigma(2, 0) = XXFad[4];
  sigma(2, 1) = XXFad[3];
  sigma(2, 2) = XXFad[2];

  alpha(0, 0) = XXFad[6];
  alpha(0, 1) = XXFad[10];
  alpha(0, 2) = XXFad[9];
  alpha(1, 0) = XXFad[10];
  alpha(1, 1) = XXFad[7];
  alpha(1, 2) = XXFad[8];
  alpha(2, 0) = XXFad[9];
  alpha(2, 1) = XXFad[8];
  alpha(2, 2) = -XXFad[6] - XXFad[7];

  DFadType kappa = XXFad[11];

  DFadType f = compute_f(sigma, alpha, kappa);

  ScalarT dfdkappa;

  dfdkappa = f.dx(11);

  return dfdkappa;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitModel<EvalT, Traits>::ScalarT>
// minitensor::Tensor<typename EvalT::ScalarT>
CapImplicitModel<EvalT, Traits>::compute_dgdsigma(
    std::vector<ScalarT> const& XX)
{
  std::vector<DFadType> XXFad(13);
  std::vector<ScalarT>  XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XX[i]);
    XXFad[i] = DFadType(13, i, XXtmp[i]);
  }

  minitensor::Tensor<DFadType> sigma(3), alpha(3);

  sigma(0, 0) = XXFad[0];
  sigma(0, 1) = XXFad[5];
  sigma(0, 2) = XXFad[4];
  sigma(1, 0) = XXFad[5];
  sigma(1, 1) = XXFad[1];
  sigma(1, 2) = XXFad[3];
  sigma(2, 0) = XXFad[4];
  sigma(2, 1) = XXFad[3];
  sigma(2, 2) = XXFad[2];

  // NOTE: DFadType alpha and kappa may not be necessary
  // since we only compute dfdsigma while keeping alpha and kappa constant
  alpha(0, 0) = XXFad[6];
  alpha(0, 1) = XXFad[10];
  alpha(0, 2) = XXFad[9];
  alpha(1, 0) = XXFad[10];
  alpha(1, 1) = XXFad[7];
  alpha(1, 2) = XXFad[8];
  alpha(2, 0) = XXFad[9];
  alpha(2, 1) = XXFad[8];
  alpha(2, 2) = -XXFad[6] - XXFad[7];

  DFadType kappa = XXFad[11];

  DFadType g = compute_g(sigma, alpha, kappa);

  minitensor::Tensor<ScalarT> dgdsigma(3);

  dgdsigma(0, 0) = g.dx(0);
  dgdsigma(0, 1) = g.dx(5);
  dgdsigma(0, 2) = g.dx(4);
  dgdsigma(1, 0) = g.dx(5);
  dgdsigma(1, 1) = g.dx(1);
  dgdsigma(1, 2) = g.dx(3);
  dgdsigma(2, 0) = g.dx(4);
  dgdsigma(2, 1) = g.dx(3);
  dgdsigma(2, 2) = g.dx(2);

  return dgdsigma;
}

template <typename EvalT, typename Traits>
minitensor::Tensor<typename CapImplicitModel<EvalT, Traits>::DFadType>
// minitensor::Tensor<typename EvalT::DFadType>
CapImplicitModel<EvalT, Traits>::compute_dgdsigma(
    std::vector<DFadType> const& XX)
{
  std::vector<D2FadType> D2XX(13);
  std::vector<DFadType>  XXFadtmp(13);
  std::vector<ScalarT>   XXtmp(13);

  for (int i = 0; i < 13; ++i) {
    XXtmp[i]    = Sacado::ScalarValue<ScalarT>::eval(XX[i].val());
    XXFadtmp[i] = DFadType(13, i, XXtmp[i]);
    D2XX[i]     = D2FadType(13, i, XXFadtmp[i]);
  }

  minitensor::Tensor<D2FadType> sigma(3), alpha(3);

  sigma(0, 0) = D2XX[0];
  sigma(0, 1) = D2XX[5];
  sigma(0, 2) = D2XX[4];
  sigma(1, 0) = D2XX[5];
  sigma(1, 1) = D2XX[1];
  sigma(1, 2) = D2XX[3];
  sigma(2, 0) = D2XX[4];
  sigma(2, 1) = D2XX[3];
  sigma(2, 2) = D2XX[2];

  // NOTE: D2FadType alpha and kappa may not be necessary
  // since we only compute dfdsigma while keeping alpha and kappa constant
  alpha(0, 0) = D2XX[6];
  alpha(0, 1) = D2XX[10];
  alpha(0, 2) = D2XX[9];
  alpha(1, 0) = D2XX[10];
  alpha(1, 1) = D2XX[7];
  alpha(1, 2) = D2XX[8];
  alpha(2, 0) = D2XX[9];
  alpha(2, 1) = D2XX[8];
  alpha(2, 2) = -D2XX[6] - D2XX[7];

  D2FadType kappa = D2XX[11];

  D2FadType g = compute_g(sigma, alpha, kappa);

  minitensor::Tensor<DFadType> dgdsigma(3);

  dgdsigma(0, 0) = g.dx(0);
  dgdsigma(0, 1) = g.dx(5);
  dgdsigma(0, 2) = g.dx(4);
  dgdsigma(1, 0) = g.dx(5);
  dgdsigma(1, 1) = g.dx(1);
  dgdsigma(1, 2) = g.dx(3);
  dgdsigma(2, 0) = g.dx(4);
  dgdsigma(2, 1) = g.dx(3);
  dgdsigma(2, 2) = g.dx(2);

  return dgdsigma;
}

//--------------------------- hardening functions ----------------------------//
template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitModel<EvalT, Traits>::compute_Galpha(T J2_alpha)
{
  if (N != 0)
    return 1.0 - pow(J2_alpha, 0.5) / N;
  else
    return 0.0;
}

template <typename EvalT, typename Traits>
template <typename T>
minitensor::Tensor<T>
CapImplicitModel<EvalT, Traits>::compute_halpha(
    minitensor::Tensor<T> const& dgdsigma,
    T const                      J2_alpha)
{
  T Galpha = compute_Galpha(J2_alpha);

  T I1 = minitensor::trace(dgdsigma), p = I1 / 3.0;

  minitensor::Tensor<T> s = dgdsigma - p * minitensor::identity<T>(3);

  // minitensor::Tensor<T, 3> halpha = calpha * Galpha * s; // * operator not
  // defined;
  minitensor::Tensor<T> halpha(3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { halpha(i, j) = calpha * Galpha * s(i, j); }
  }

  return halpha;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitModel<EvalT, Traits>::compute_dedkappa(T const kappa)
{
  //******** use analytical expression
  T Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;

  T X0 = kappa0 - Q * Ff_kappa0;

  T Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

  T X = kappa - Q * Ff_kappa;

  T dedX =
      (D1 - 2. * D2 * (X - X0)) * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

  T dXdkappa = 1. + Q * C * L * exp(L * kappa) + Q * phi;

  return dedX * dXdkappa;
}

template <typename EvalT, typename Traits>
template <typename T>
T
CapImplicitModel<EvalT, Traits>::compute_hkappa(
    T const I1_dgdsigma,
    T const dedkappa)
{
  if (dedkappa != 0)
    return I1_dgdsigma / dedkappa;
  else
    return 0;
}

//------------------------ elasto-plastic tangent modulus --------------------//
template <typename EvalT, typename Traits>
minitensor::Tensor4<typename CapImplicitModel<EvalT, Traits>::ScalarT>
CapImplicitModel<EvalT, Traits>::compute_Cep(
    minitensor::Tensor4<ScalarT>& Celastic,
    minitensor::Tensor<ScalarT>&  sigma,
    minitensor::Tensor<ScalarT>&  alpha,
    ScalarT&                      kappa,
    ScalarT&                      dgamma)
{
  if (dgamma == 0) return Celastic;

  // define variable
  minitensor::Tensor4<ScalarT> Cep(num_dims_);

  std::vector<ScalarT> XX(13);

  minitensor::Tensor<ScalarT> dfdsigma;
  minitensor::Tensor<ScalarT> dfdalpha;
  minitensor::Tensor<ScalarT> dgdsigma;
  minitensor::Tensor<ScalarT> halpha;
  ScalarT                     hkappa;
  ScalarT                     dfdkappa;
  ScalarT                     chi;

  // compute variable
  XX = initialize(sigma, alpha, kappa, dgamma);

  dfdsigma = compute_dfdsigma(XX);
  dfdalpha = dfdsigma * (-1.0);
  dfdkappa = compute_dfdkappa(XX);
  dgdsigma = compute_dgdsigma(XX);

  ScalarT J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);
  halpha           = compute_halpha(dgdsigma, J2_alpha);

  ScalarT I1_dgdsigma = minitensor::trace(dgdsigma);
  ScalarT dedkappa    = compute_dedkappa(kappa);
  hkappa              = compute_hkappa(I1_dgdsigma, dedkappa);

  chi = minitensor::dotdot(minitensor::dotdot(dfdsigma, Celastic), dgdsigma) -
        minitensor::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

  if (chi == 0) {
    std::cout << "Chi equals to 0 error during computing elasto-plastic tangent"
              << std::endl;
    chi = 1e-16;
  }

  // compute tangent
  Cep =
      Celastic - 1.0 / chi *
                     minitensor::dotdot(
                         Celastic,
                         minitensor::tensor(
                             dgdsigma, minitensor::dotdot(dfdsigma, Celastic)));

  return Cep;
}  // end compute tangent function

//-------------------- elasto-plastic perfect tangent modulus ----------------//
template <typename EvalT, typename Traits>
minitensor::Tensor4<typename CapImplicitModel<EvalT, Traits>::ScalarT>
CapImplicitModel<EvalT, Traits>::compute_Cepp(
    minitensor::Tensor4<ScalarT>& Celastic,
    minitensor::Tensor<ScalarT>&  sigma,
    minitensor::Tensor<ScalarT>&  alpha,
    ScalarT&                      kappa,
    ScalarT&                      dgamma)
{
  if (dgamma == 0) return Celastic;

  // define variable
  minitensor::Tensor4<ScalarT> Cepp(num_dims_);

  std::vector<ScalarT> XX(13);

  minitensor::Tensor<ScalarT> dfdsigma;
  minitensor::Tensor<ScalarT> dfdalpha;
  minitensor::Tensor<ScalarT> dgdsigma;
  minitensor::Tensor<ScalarT> halpha;
  ScalarT                     hkappa;
  ScalarT                     dfdkappa;
  ScalarT                     chi;

  // compute variable
  XX = initialize(sigma, alpha, kappa, dgamma);

  dfdsigma = compute_dfdsigma(XX);
  dfdalpha = dfdsigma * (-1.0);
  dfdkappa = compute_dfdkappa(XX);
  dgdsigma = compute_dgdsigma(XX);

  ScalarT J2_alpha = 0.5 * minitensor::dotdot(alpha, alpha);
  halpha           = compute_halpha(dgdsigma, J2_alpha);

  ScalarT I1_dgdsigma = minitensor::trace(dgdsigma);
  ScalarT dedkappa    = compute_dedkappa(kappa);
  hkappa              = compute_hkappa(I1_dgdsigma, dedkappa);

  chi = minitensor::dotdot(minitensor::dotdot(dfdsigma, Celastic), dgdsigma);

  if (chi == 0) {
    std::cout << "Chi equals to 0 error during computing elasto-plastic tangent"
              << std::endl;
    chi = 1e-16;
  }

  // compute tangent
  Cepp =
      Celastic - 1.0 / chi *
                     minitensor::dotdot(
                         Celastic,
                         minitensor::tensor(
                             dgdsigma, minitensor::dotdot(dfdsigma, Celastic)));

  return Cepp;
}  // end compute tangent function

}  // namespace LCM
