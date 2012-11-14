//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  CapExplicit<EvalT, Traits>::CapExplicit(const Teuchos::ParameterList& p) :
      elasticModulus(p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), poissonsRatio(
          p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), strain(
          p.get<std::string>("Strain Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), backStress(
          p.get<std::string>("Back Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), capParameter(
          p.get<std::string>("Cap Parameter Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), friction(
          p.get<std::string>("Friction Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), dilatancy(
          p.get<std::string>("Dilatancy Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), eqps(
          p.get<std::string>("Eqps Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), hardeningModulus(
          p.get<std::string>("Hardening Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), A(
          p.get<RealType>("A Name")), B(p.get<RealType>("B Name")), C(
          p.get<RealType>("C Name")), theta(p.get<RealType>("Theta Name")), R(
          p.get<RealType>("R Name")), kappa0(p.get<RealType>("Kappa0 Name")), W(
          p.get<RealType>("W Name")), D1(p.get<RealType>("D1 Name")), D2(
          p.get<RealType>("D2 Name")), calpha(p.get<RealType>("Calpha Name")), psi(
          p.get<RealType>("Psi Name")), N(p.get<RealType>("N Name")), L(
          p.get<RealType>("L Name")), phi(p.get<RealType>("Phi Name")), Q(
          p.get<RealType>("Q Name"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];

    this->addDependentField(elasticModulus);
    // PoissonRatio not used in 1D stress calc
    if (numDims > 1) this->addDependentField(poissonsRatio);
    this->addDependentField(strain);

    // state variable
    strainName = p.get<std::string>("Strain Name") + "_old";
    stressName = p.get<std::string>("Stress Name") + "_old";
    backStressName = p.get<std::string>("Back Stress Name") + "_old";
    capParameterName = p.get<std::string>("Cap Parameter Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";

    // evaluated fields
    this->addEvaluatedField(stress);
    this->addEvaluatedField(backStress);
    this->addEvaluatedField(capParameter);
    this->addEvaluatedField(friction);
    this->addEvaluatedField(dilatancy);
    this->addEvaluatedField(eqps);
    this->addEvaluatedField(hardeningModulus);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void CapExplicit<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(elasticModulus, fm);
    if (numDims > 1) this->utils.setFieldData(poissonsRatio, fm);
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(strain, fm);
    this->utils.setFieldData(backStress, fm);
    this->utils.setFieldData(capParameter, fm);
    this->utils.setFieldData(friction, fm);
    this->utils.setFieldData(dilatancy, fm);
    this->utils.setFieldData(eqps, fm);
    this->utils.setFieldData(hardeningModulus, fm);
  }

//**********************************************************************

  template<typename EvalT, typename Traits>
  void CapExplicit<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {

    // previous state
    Albany::MDArray strainold = (*workset.stateArrayPtr)[strainName];
    Albany::MDArray stressold = (*workset.stateArrayPtr)[stressName];
    Albany::MDArray backStressold = (*workset.stateArrayPtr)[backStressName];
    Albany::MDArray capParameterold = (*workset.stateArrayPtr)[capParameterName];
    Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // local parameters
        ScalarT lame = elasticModulus(cell, qp) * poissonsRatio(cell, qp)
            / (1.0 + poissonsRatio(cell, qp))
            / (1.0 - 2.0 * poissonsRatio(cell, qp));
        ScalarT mu = elasticModulus(cell, qp) / 2.0
            / (1.0 + poissonsRatio(cell, qp));
        ScalarT bulkModulus = lame + (2. / 3.) * mu;

        // elastic matrix
        LCM::Tensor4<ScalarT> Celastic = lame * LCM::identity_3<ScalarT>(3)
            + mu * (LCM::identity_1<ScalarT>(3) + LCM::identity_2<ScalarT>(3));

        // elastic compliance tangent matrix
        LCM::Tensor4<ScalarT> compliance =
            (1. / bulkModulus / 9.) * LCM::identity_3<ScalarT>(3)
                + (1. / mu / 2.)
                    * (0.5
                        * (LCM::identity_1<ScalarT>(3)
                            + LCM::identity_2<ScalarT>(3))
                        - (1. / 3.) * LCM::identity_3<ScalarT>(3));

        // incremental strain tensor
        LCM::Tensor<ScalarT> depsilon(3);
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);

        // trial state
        LCM::Tensor<ScalarT> sigmaVal = LCM::dotdot(Celastic, depsilon);
        LCM::Tensor<ScalarT> alphaVal = LCM::identity<ScalarT>(3);
        LCM::Tensor<ScalarT> sigmaN(3), strainN(3); // previous state

        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            sigmaN(i, j) = stressold(cell, qp, i, j);
            strainN(i, j) = strainold(cell, qp, i, j);
            sigmaVal(i, j) = sigmaVal(i, j) + stressold(cell, qp, i, j);
            alphaVal(i, j) = backStressold(cell, qp, i, j);
          }
        }

        ScalarT kappaVal = capParameterold(cell, qp);

        // initialize friction and dilatancy (which will be updated only if plasticity occurs)
        friction(cell, qp) = 0.0;
        dilatancy(cell, qp) = 0.0;
        hardeningModulus(cell, qp) = 0.0;

        // define generalized plastic hardening modulus H
        ScalarT H(0.0), Htan(0.0);

        // define plastic strain increment, its two invariants: dev, and vol
        LCM::Tensor<ScalarT> deps_plastic(3, 0.0);
        ScalarT deqps(0.0), devolps(0.0);

        // define a temporary tensor to store trial stress tensors
        LCM::Tensor<ScalarT> sigmaTr = sigmaVal;
        // define a temporary tensor to store previous back stress
        LCM::Tensor<ScalarT> alphaN = alphaVal;

        // check yielding
        ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);

        // plastic correction
        ScalarT dgamma = 0.0;
        if (f > 1.0e-10) {
          LCM::Tensor<ScalarT> dfdsigma = compute_dfdsigma(sigmaN, alphaVal,
              kappaVal);

          LCM::Tensor<ScalarT> dgdsigma = compute_dgdsigma(sigmaN, alphaVal,
              kappaVal);

          LCM::Tensor<ScalarT> dfdalpha = -dfdsigma;

          ScalarT dfdkappa = compute_dfdkappa(sigmaN, alphaVal, kappaVal);

          ScalarT J2_alpha = 0.5 * LCM::dotdot(alphaVal, alphaVal);

          LCM::Tensor<ScalarT> halpha = compute_halpha(dgdsigma, J2_alpha);

          ScalarT I1_dgdsigma = LCM::trace(dgdsigma);

          ScalarT dedkappa = compute_dedkappa(kappaVal);

          ScalarT hkappa;
          if (dedkappa != 0)
            hkappa = I1_dgdsigma / dedkappa;
          else
            hkappa = 0;

          ScalarT kai(0.0);
          kai = LCM::dotdot(dfdsigma, LCM::dotdot(Celastic, dgdsigma))
              - LCM::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          H = -LCM::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          LCM::Tensor<ScalarT> dfdotCe = LCM::dotdot(dfdsigma, Celastic);

          if (kai != 0)
            dgamma = LCM::dotdot(dfdotCe, depsilon) / kai;
          else
            dgamma = 0;

          // update
          sigmaVal -= dgamma * LCM::dotdot(Celastic, dgdsigma);

          alphaVal += dgamma * halpha;

          // restrictions on kappa, only allow monotonic decreasing (cap hardening)
          ScalarT dkappa = dgamma * hkappa;
          if (dkappa > 0) {
            dkappa = 0;
            H = -LCM::dotdot(dfdalpha, halpha);
          }

          kappaVal += dkappa;

          // stress correction algorithm to avoid drifting from yield surface
          bool condition = false;
          int iteration = 0;
          while (condition == false) {
            f = compute_f(sigmaVal, alphaVal, kappaVal);

            LCM::Tensor<ScalarT> dfdsigma = compute_dfdsigma(sigmaVal, alphaVal,
                kappaVal);

            LCM::Tensor<ScalarT> dgdsigma = compute_dgdsigma(sigmaVal, alphaVal,
                kappaVal);

            LCM::Tensor<ScalarT> dfdalpha = -dfdsigma;

            ScalarT dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);

            J2_alpha = 0.5 * LCM::dotdot(alphaVal, alphaVal);

            halpha = compute_halpha(dgdsigma, J2_alpha);

            I1_dgdsigma = LCM::trace(dgdsigma);

            dedkappa = compute_dedkappa(kappaVal);

            if (dedkappa != 0)
              hkappa = I1_dgdsigma / dedkappa;
            else
              hkappa = 0;

            //generalized plastic hardening modulus
            H = -LCM::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

            kai = LCM::dotdot(dfdsigma, LCM::dotdot(Celastic, dgdsigma));
            kai = kai - LCM::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

            if (std::abs(f) < 1.0e-10) break;
            if (iteration > 20) {
              // output for debug
              //std::cout << "no stress correction after iteration = "
              //<< iteration << " yield function abs(f) = " << abs(f)
              //<< std::endl;
              break;
            }

            ScalarT delta_gamma;
            if (kai != 0)
              delta_gamma = f / kai;
            else
              delta_gamma = 0;

            LCM::Tensor<ScalarT> sigmaK(3, 0.0), alphaK(3, 0.0);
            ScalarT kappaK(0.0);

            // restrictions on kappa, only allow monotonic decreasing
            dkappa = delta_gamma * hkappa;
            if (dkappa > 0) {
              dkappa = delta_gamma * 0.0;
              H = -LCM::dotdot(dfdalpha, halpha);
            }

            sigmaK = sigmaVal - delta_gamma * LCM::dotdot(Celastic, dgdsigma);
            alphaK = alphaVal + delta_gamma * halpha;
            kappaK = kappaVal + dkappa;

            ScalarT fpre = compute_f(sigmaK, alphaK, kappaK);

            if (std::abs(fpre) > std::abs(f)) {
              // if the corrected stress is further away from yield surface, then use normal correction
              ScalarT dfdotdf = LCM::dotdot(dfdsigma, dfdsigma);
              if (dfdotdf != 0)
                delta_gamma = f / dfdotdf;
              else
                delta_gamma = 0.0;

              sigmaK = sigmaVal - delta_gamma * dfdsigma;
              alphaK = alphaVal;
              kappaK = kappaVal;

              H = 0.0;

            }

            sigmaVal = sigmaK;
            alphaVal = alphaK;
            kappaVal = kappaK;

            iteration++;

          } // end of stress correction

          //compute plastic strain increment deps_plastic = compliance ( sigma_tr - sigma_(n+1));
          LCM::Tensor<ScalarT> dsigma = sigmaTr - sigmaVal;
          deps_plastic = LCM::dotdot(compliance, dsigma);

          // compute its two invariants: devolps (volumetric) and deqps (deviatoric)
          devolps = LCM::trace(deps_plastic);
          LCM::Tensor<ScalarT> dev_plastic = deps_plastic
              - (1. / 3.) * devolps * LCM::identity<ScalarT>(3);
          //deqps = std::sqrt(2./3.) * LCM::norm(dev_plastic);
          // use altenative definition, just differ by constants
          deqps = std::sqrt(2) * LCM::norm(dev_plastic);

          // dilatancy
          if (deqps != 0)
            dilatancy(cell, qp) = devolps / deqps;
          else
            dilatancy(cell, qp) = 0.0;

          // previous p and tau
          ScalarT pN(0.0), tauN(0.0);
          LCM::Tensor<ScalarT> xi = sigmaN - alphaN;
          pN = LCM::trace(xi);
          pN = pN / 3.;
          LCM::Tensor<ScalarT> sN = xi - pN * LCM::identity<ScalarT>(3);
          //qN = sqrt(3./2.) * LCM::norm(sN);
          tauN = sqrt(1. / 2.) * LCM::norm(sN);

          // current p, and tau
          ScalarT p(0.0), tau(0.0);
          xi = sigmaVal - alphaVal;
          p = LCM::trace(xi);
          p = p / 3.;
          LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>(3);
          //q = sqrt(3./2.) * LCM::norm(s);
          tau = sqrt(1. / 2.) * LCM::norm(s);
          //LCM::Tensor<ScalarT, 3> ds = s - sN;

          // difference
          ScalarT dtau = tau - tauN;
          //ScalarT dtau = sqrt(1./2.) * LCM::norm(ds);
          ScalarT dp = p - pN;

          // friction coefficient by finite difference
          if (dp != 0)
            friction(cell, qp) = dtau / dp;
          else
            friction(cell, qp) = 0.0;

          // previous r(gamma)
          ScalarT rN(0.0);
          ScalarT evol3 = LCM::trace(strainN);
          evol3 = evol3 / 3.;
          LCM::Tensor<ScalarT> e = strainN - evol3 * LCM::identity<ScalarT>(3);
          rN = sqrt(2.) * LCM::norm(e);

          // current r(gamma)
          ScalarT r(0.0);
          LCM::Tensor<ScalarT> strainCurrent = strainN + depsilon;
          evol3 = LCM::trace(strainCurrent);
          evol3 = evol3 / 3.;
          e = strainCurrent - evol3 * LCM::identity<ScalarT>(3);
          r = sqrt(2.) * LCM::norm(e);

          // difference
          ScalarT dr = r - rN;
          // tagent hardening modulus
          if (dr != 0) Htan = dtau / dr;

          if (std::abs(1. - Htan / mu) > 1.0e-10)
            hardeningModulus(cell, qp) = Htan / (1. - Htan / mu);
          else
            hardeningModulus(cell, qp) = 0.0;

        } // end of plastic correction

        // output for debugging
//        std::cout << "friction = " << Sacado::ScalarValue<ScalarT>::eval(friction(cell,qp)) << std::endl;
//        std::cout << "dilatancy = " << Sacado::ScalarValue<ScalarT>::eval(dilatancy(cell,qp)) << std::endl;
//        std::cout << "hardeningModulus = " << Sacado::ScalarValue<ScalarT>::eval(hardeningModulus(cell,qp)) << std::endl;
//        std::cout << "============="<< std::endl;

        // update
        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = sigmaVal(i, j);
            backStress(cell, qp, i, j) = alphaVal(i, j);
          }
        }

        capParameter(cell, qp) = kappaVal;
        eqps(cell, qp) = eqpsold(cell, qp) + deqps;

      } //loop over qps

    } //loop over cell

  } // end of evaluateFields

//**********************************************************************
// all local functions
  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT CapExplicit<EvalT, Traits>::compute_f(
      LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha,
      ScalarT & kappa)
  {

    LCM::Tensor<ScalarT> xi = sigma - alpha;

    ScalarT I1 = LCM::trace(xi), p = I1 / 3;

    LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * LCM::dotdot(s, s);

    ScalarT J3 = LCM::det(s);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0) Gamma = 0.5
        * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
            + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0)) Fc = 1.0
        - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
  }

  template<typename EvalT, typename Traits>
  LCM::Tensor<typename CapExplicit<EvalT, Traits>::ScalarT> CapExplicit<
      EvalT, Traits>::compute_dfdsigma(LCM::Tensor<ScalarT> & sigma,
      LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
  {
    LCM::Tensor<ScalarT> dfdsigma(3);

    LCM::Tensor<ScalarT> xi = sigma - alpha;

    ScalarT I1 = LCM::trace(xi), p = I1 / 3;

    LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * LCM::dotdot(s, s);

    ScalarT J3 = LCM::det(s);

    LCM::Tensor<ScalarT> id = LCM::identity<ScalarT>(3);
    LCM::Tensor<ScalarT> dI1dsigma = id;
    LCM::Tensor<ScalarT> dJ2dsigma = s;
    LCM::Tensor<ScalarT> dJ3dsigma(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * id(i, j) / 3;

    ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0) Fc = 1.0
        - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0) Gamma = 0.5
        * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
            + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    // derivatives
    ScalarT dFfdI1 = -(B * C * std::exp(B * I1) + theta);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0)) dFcdI1 = -2.0 * (I1 - kappa)
        / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0) dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0
        * (1.0 - 1.0 / psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0) dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0
        * (1.0 - 1.0 / psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    dfdsigma = dfdI1 * dI1dsigma + dfdJ2 * dJ2dsigma + dfdJ3 * dJ3dsigma;

    return dfdsigma;
  }

  template<typename EvalT, typename Traits>
  LCM::Tensor<typename CapExplicit<EvalT, Traits>::ScalarT> CapExplicit<
      EvalT, Traits>::compute_dgdsigma(LCM::Tensor<ScalarT> & sigma,
      LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
  {
    LCM::Tensor<ScalarT> dgdsigma(3);

    LCM::Tensor<ScalarT> xi = sigma - alpha;

    ScalarT I1 = LCM::trace(xi), p = I1 / 3;

    LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * LCM::dotdot(s, s);

    ScalarT J3 = LCM::det(s);

    LCM::Tensor<ScalarT> id = LCM::identity<ScalarT>(3);
    LCM::Tensor<ScalarT> dI1dsigma = id;
    LCM::Tensor<ScalarT> dJ2dsigma = s;
    LCM::Tensor<ScalarT> dJ3dsigma(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        dJ3dsigma(i, j) = s(i, j) * s(i, j) - 2 * J2 * id(i, j) / 3;

    ScalarT Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

    ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

    ScalarT X = kappa - Q * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0) Fc = 1.0
        - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0) Gamma = 0.5
        * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
            + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    // derivatives
    ScalarT dFfdI1 = -(L * C * std::exp(L * I1) + phi);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0)) dFcdI1 = -2.0 * (I1 - kappa)
        / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0) dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0
        * (1.0 - 1.0 / psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0) dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0
        * (1.0 - 1.0 / psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    dgdsigma = dfdI1 * dI1dsigma + dfdJ2 * dJ2dsigma + dfdJ3 * dJ3dsigma;

    return dgdsigma;
  }

  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT CapExplicit<EvalT, Traits>::compute_dfdkappa(
      LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha,
      ScalarT & kappa)
  {
    ScalarT dfdkappa;
    LCM::Tensor<ScalarT> dfdsigma(3);

    LCM::Tensor<ScalarT> xi = sigma - alpha;

    ScalarT I1 = LCM::trace(xi), p = I1 / 3;

    LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * LCM::dotdot(s, s);

    ScalarT J3 = LCM::det(s);

    ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT dFcdkappa = 0.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0)) dFcdkappa = 2 * (I1 - kappa)
        * ((X - kappa)
            + R * (I1 - kappa) * (theta + B * C * std::exp(B * kappa)))
        / (X - kappa) / (X - kappa) / (X - kappa);

    dfdkappa = -dFcdkappa * (Ff_I1 - N) * (Ff_I1 - N);

    return dfdkappa;
  }
  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT CapExplicit<EvalT, Traits>::compute_Galpha(
      ScalarT & J2_alpha)
  {
    if (N != 0)
      return 1.0 - std::pow(J2_alpha, 0.5) / N;
    else
      return 0.0;
  }

  template<typename EvalT, typename Traits>
  LCM::Tensor<typename CapExplicit<EvalT, Traits>::ScalarT> CapExplicit<
      EvalT, Traits>::compute_halpha(LCM::Tensor<ScalarT> & dgdsigma,
      ScalarT & J2_alpha)
  {

    ScalarT Galpha = compute_Galpha(J2_alpha);

    ScalarT I1 = LCM::trace(dgdsigma), p = I1 / 3;

    LCM::Tensor<ScalarT> s = dgdsigma - p * LCM::identity<ScalarT>(3);

    LCM::Tensor<ScalarT> halpha(3);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        halpha(i, j) = calpha * Galpha * s(i, j);
      }
    }

    return halpha;
  }

  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT CapExplicit<EvalT, Traits>::compute_dedkappa(
      ScalarT & kappa)
  {
    ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;

    ScalarT X0 = kappa0 - Q * Ff_kappa0;

    ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

    ScalarT X = kappa - Q * Ff_kappa;

    ScalarT dedX = (D1 - 2 * D2 * (X - X0))
        * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

    ScalarT dXdkappa = 1 + Q * C * L * std::exp(L * kappa) + Q * phi;

    return dedX * dXdkappa;
  }

//**********************************************************************
}// end LCM
