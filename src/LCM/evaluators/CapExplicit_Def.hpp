//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM
{

//**********************************************************************
  template<typename EvalT, typename Traits>
  CapExplicit<EvalT, Traits>::
  CapExplicit(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
    elasticModulus(p.get<std::string>("Elastic Modulus Name"),dl->qp_scalar),
    poissonsRatio(p.get<std::string>("Poissons Ratio Name"),dl->qp_scalar),
    strain(p.get<std::string>("Strain Name"),dl->qp_tensor),
    stress(p.get<std::string>("Stress Name"),dl->qp_tensor),
    backStress(p.get<std::string>("Back Stress Name"),dl->qp_tensor),
    capParameter(p.get<std::string>("Cap Parameter Name"),dl->qp_scalar),
    //friction(p.get<std::string>("Friction Name"),dl->qp_scalar),
    //dilatancy(p.get<std::string>("Dilatancy Name"),dl->qp_scalar),
    eqps(p.get<std::string>("Eqps Name"),dl->qp_scalar),
    //hardeningModulus(p.get<std::string>("Hardening Modulus Name"),dl->qp_scalar),
    volPlasticStrain(p.get<std::string>("Vol Plastic Strain Name"),dl->qp_scalar),
    A(p.get<RealType>("A Name")),
    B(p.get<RealType>("B Name")),
    C(p.get<RealType>("C Name")),
    theta(p.get<RealType>("Theta Name")),
    R(p.get<RealType>("R Name")),
    kappa0(p.get<RealType>("Kappa0 Name")),
    W(p.get<RealType>("W Name")),
    D1(p.get<RealType>("D1 Name")),
    D2(p.get<RealType>("D2 Name")),
    calpha(p.get<RealType>("Calpha Name")),
    psi(p.get<RealType>("Psi Name")),
    N(p.get<RealType>("N Name")),
    L(p.get<RealType>("L Name")),
    phi(p.get<RealType>("Phi Name")),
    Q(p.get<RealType>("Q Name"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];

    this->addDependentField(elasticModulus);
    // PoissonRatio not used in 1D stress calc
    if (numDims > 1)
      this->addDependentField(poissonsRatio);
    this->addDependentField(strain);

    // state variable
    strainName = p.get<std::string>("Strain Name") + "_old";
    stressName = p.get<std::string>("Stress Name") + "_old";
    backStressName = p.get<std::string>("Back Stress Name") + "_old";
    capParameterName = p.get<std::string>("Cap Parameter Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";
    volPlasticStrainName = p.get<std::string>("Vol Plastic Strain Name")
        + "_old";

    // evaluated fields
    this->addEvaluatedField(stress);
    this->addEvaluatedField(backStress);
    this->addEvaluatedField(capParameter);
    //this->addEvaluatedField(friction);
    //this->addEvaluatedField(dilatancy);
    this->addEvaluatedField(eqps);
    //this->addEvaluatedField(hardeningModulus);
    this->addEvaluatedField(volPlasticStrain);

    this->setName("Stress" + PHX::typeAsString<EvalT>());

    // initialize tensor
    I = Intrepid2::eye<ScalarT>(numDims);
    id1 = Intrepid2::identity_1<ScalarT>(numDims);
    id2 = Intrepid2::identity_2<ScalarT>(numDims);
    id3 = Intrepid2::identity_3<ScalarT>(numDims);
    Celastic = Intrepid2::Tensor4<ScalarT>(numDims);
    compliance = Intrepid2::Tensor4<ScalarT>(numDims);
    depsilon = Intrepid2::Tensor<ScalarT>(numDims);
    sigmaN = Intrepid2::Tensor<ScalarT>(numDims);
    strainN = Intrepid2::Tensor<ScalarT>(numDims);
    sigmaVal = Intrepid2::Tensor<ScalarT>(numDims);
    alphaVal = Intrepid2::Tensor<ScalarT>(numDims);
    deps_plastic = Intrepid2::Tensor<ScalarT>(numDims);
    sigmaTr = Intrepid2::Tensor<ScalarT>(numDims);
    alphaTr = Intrepid2::Tensor<ScalarT>(numDims);
    dfdsigma = Intrepid2::Tensor<ScalarT>(numDims);
    dgdsigma = Intrepid2::Tensor<ScalarT>(numDims);
    dfdalpha = Intrepid2::Tensor<ScalarT>(numDims);
    halpha = Intrepid2::Tensor<ScalarT>(numDims);
    dfdotCe = Intrepid2::Tensor<ScalarT>(numDims);
    sigmaK = Intrepid2::Tensor<ScalarT>(numDims);
    alphaK = Intrepid2::Tensor<ScalarT>(numDims);
    dsigma = Intrepid2::Tensor<ScalarT>(numDims);
    dev_plastic = Intrepid2::Tensor<ScalarT>(numDims);
    xi = Intrepid2::Tensor<ScalarT>(numDims);
    sN = Intrepid2::Tensor<ScalarT>(numDims);
    s = Intrepid2::Tensor<ScalarT>(numDims);
    strainCurrent = Intrepid2::Tensor<ScalarT>(numDims);
    dJ3dsigma = Intrepid2::Tensor<ScalarT>(numDims);
    eps_dev = Intrepid2::Tensor<ScalarT>(numDims);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void CapExplicit<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(elasticModulus, fm);
    if (numDims > 1)
      this->utils.setFieldData(poissonsRatio, fm);
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(strain, fm);
    this->utils.setFieldData(backStress, fm);
    this->utils.setFieldData(capParameter, fm);
    //this->utils.setFieldData(friction, fm);
    //this->utils.setFieldData(dilatancy, fm);
    this->utils.setFieldData(eqps, fm);
    //this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(volPlasticStrain, fm);

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
    Albany::MDArray volPlasticStrainold =
        (*workset.stateArrayPtr)[volPlasticStrainName];

    ScalarT lame, mu, bulkModulus;
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // local parameters
        lame = elasticModulus(cell, qp) * poissonsRatio(cell, qp)
            / (1.0 + poissonsRatio(cell, qp))
            / (1.0 - 2.0 * poissonsRatio(cell, qp));
        mu = elasticModulus(cell, qp) / 2.0
            / (1.0 + poissonsRatio(cell, qp));
        bulkModulus = lame + (2. / 3.) * mu;

        // elastic matrix
        Celastic = lame * id3 + mu * (id1 + id2);

        // elastic compliance tangent matrix
        compliance = (1. / bulkModulus / 9.) * id3
            + (1. / mu / 2.) * (0.5 * (id1 + id2)- (1. / 3.) * id3);

        // trial state
        Intrepid2::Tensor<ScalarT> depsilon(3);
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);
            strainN(i, j) = strainold(cell, qp, i, j);
            sigmaN(i,j) = stressold(cell,qp,i,j);
            alphaVal(i,j) = backStressold(cell,qp,i,j);
          }
        }

        sigmaVal = sigmaN + Intrepid2::dotdot(Celastic, depsilon);
        ScalarT kappaVal = capParameterold(cell, qp);

        // initialize friction and dilatancy
        // (which will be updated only if plasticity occurs)
        //friction(cell, qp) = 0.0;
        //dilatancy(cell, qp) = 0.0;
        //hardeningModulus(cell, qp) = 0.0;

        // define generalized plastic hardening modulus H
        //ScalarT H(0.0), Htan(0.0);

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

          ScalarT J2_alpha = 0.5 * Intrepid2::dotdot(alphaVal, alphaVal);

          halpha = compute_halpha(dgdsigma, J2_alpha);

          ScalarT I1_dgdsigma = Intrepid2::trace(dgdsigma);

          ScalarT dedkappa = compute_dedkappa(kappaVal);

          ScalarT hkappa;
          if (dedkappa != 0.0)
            hkappa = I1_dgdsigma / dedkappa;
          else
            hkappa = 0.0;

          ScalarT kai(0.0);
          kai = Intrepid2::dotdot(dfdsigma, Intrepid2::dotdot(Celastic, dgdsigma))
              - Intrepid2::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          //H = -Intrepid2::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

          dfdotCe = Intrepid2::dotdot(dfdsigma, Celastic);

          if (kai != 0.0)
            dgamma = Intrepid2::dotdot(dfdotCe, depsilon) / kai;
          else
            dgamma = 0.0;

          // update
          sigmaVal -= dgamma * Intrepid2::dotdot(Celastic, dgdsigma);

          alphaVal += dgamma * halpha;

          // restrictions on kappa, only allow monotonic decreasing (cap hardening)
          ScalarT dkappa = dgamma * hkappa;
          if (dkappa > 0) {
            dkappa = 0;
            //H = -Intrepid2::dotdot(dfdalpha, halpha);
          }

          kappaVal += dkappa;

          // stress correction algorithm to avoid drifting from yield surface
          bool condition = false;
          int iteration = 0;
          int max_iteration = 20;
          RealType tolerance = 1.0e-10;
          while (condition == false) {
            f = compute_f(sigmaVal, alphaVal, kappaVal);

            dfdsigma = compute_dfdsigma(sigmaVal, alphaVal, kappaVal);

            dgdsigma = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);

            dfdalpha = -dfdsigma;

            ScalarT dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);

            J2_alpha = 0.5 * Intrepid2::dotdot(alphaVal, alphaVal);

            halpha = compute_halpha(dgdsigma, J2_alpha);

            I1_dgdsigma = Intrepid2::trace(dgdsigma);

            dedkappa = compute_dedkappa(kappaVal);

            if (dedkappa != 0)
              hkappa = I1_dgdsigma / dedkappa;
            else
              hkappa = 0;

            //generalized plastic hardening modulus
            //H = -Intrepid2::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

            kai = Intrepid2::dotdot(dfdsigma,
                Intrepid2::dotdot(Celastic, dgdsigma));
            kai = kai - Intrepid2::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

            if (std::abs(f) < tolerance)
              break;
            if (iteration > max_iteration) {
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

            // restrictions on kappa, only allow monotonic decreasing
            dkappa = delta_gamma * hkappa;
            if (dkappa > 0.0) {
              dkappa = 0.0;
              //H = -Intrepid2::dotdot(dfdalpha, halpha);
            }

            // update
            sigmaK = sigmaVal
                - delta_gamma * Intrepid2::dotdot(Celastic, dgdsigma);
            alphaK = alphaVal + delta_gamma * halpha;
            ScalarT kappaK = kappaVal + dkappa;

            ScalarT fK = compute_f(sigmaK, alphaK, kappaK);

            if (std::abs(fK) > std::abs(f)) {
              // if the corrected stress is further away from yield surface,
              // then use normal correction
              ScalarT dfdotdf = Intrepid2::dotdot(dfdsigma, dfdsigma);
              if (dfdotdf != 0)
                delta_gamma = f / dfdotdf;
              else
                delta_gamma = 0.0;

              sigmaK = sigmaVal - delta_gamma * dfdsigma;
              alphaK = alphaVal;
              kappaK = kappaVal;

              //H = 0.0;
            }

            sigmaVal = sigmaK;
            alphaVal = alphaK;
            kappaVal = kappaK;

            iteration++;

          } // end of stress correction

          // compute plastic strain increment
          // deps_plastic = compliance ( sigma_tr - sigma_(n+1));
          dsigma = sigmaTr - sigmaVal;
          deps_plastic = Intrepid2::dotdot(compliance, dsigma);

          // compute its two invariants
          // devolps (volumetric) and deqps (deviatoric)
          devolps = Intrepid2::trace(deps_plastic);
          dev_plastic = deps_plastic - (1. / 3.) * devolps * I;
          // use altenative definition, differ by constants
          deqps = std::sqrt(2) * Intrepid2::norm(dev_plastic);

          // dilatancy
          //if (deqps != 0)
          //  dilatancy(cell, qp) = devolps / deqps;
          //else
          //  dilatancy(cell, qp) = 0.0;

          // previous p and tau
          //ScalarT pN(0.0), tauN(0.0);
          //xi = sigmaN - alphaTr;
          //pN = Intrepid2::trace(xi);
          //pN = pN / 3.;
          //sN = xi - pN * I;
          //tauN = sqrt(1. / 2.) * Intrepid2::norm(sN);

          // current p, and tau
          //ScalarT p(0.0), tau(0.0);
          //xi = sigmaVal - alphaVal;
          //p = Intrepid2::trace(xi);
          //p = p / 3.;
          //s = xi - p * I;
          //tau = sqrt(1. / 2.) * Intrepid2::norm(s);

          // difference
          //ScalarT dtau = tau - tauN;
          //ScalarT dp = p - pN;

          // friction coefficient by finite difference
          //if (dp != 0)
          //  friction(cell, qp) = dtau / dp;
          //else
          //  friction(cell, qp) = 0.0;

          // previous gamma(gamma)
          //ScalarT evol3 = Intrepid2::trace(strainN);
          //evol3 = evol3 / 3.;
          //eps_dev = strainN - evol3 * I;
          //ScalarT gammaN = sqrt(2.) * Intrepid2::norm(eps_dev);

          // current gamma(gamma)
          //strainCurrent = strainN + depsilon;
          //evol3 = Intrepid2::trace(strainCurrent);
          //evol3 = evol3 / 3.;
          //eps_dev = strainCurrent - evol3 * I;
          //ScalarT gamma = sqrt(2.) * Intrepid2::norm(eps_dev);

          // difference
          //ScalarT dGamma = gamma - gammaN;
          // tagent hardening modulus
          //if (dGamma != 0)
          //  Htan = dtau / dGamma;

          //if (std::abs(1. - Htan / mu) > 1.0e-10)
          //  hardeningModulus(cell, qp) = Htan / (1. - Htan / mu);
          //else
          //  hardeningModulus(cell, qp) = 0.0;

        } // end of plastic correction

        // update
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = sigmaVal(i, j);
            backStress(cell, qp, i, j) = alphaVal(i, j);
          }
        }

        capParameter(cell, qp) = kappaVal;
        eqps(cell, qp) = eqpsold(cell, qp) + deqps;
        volPlasticStrain(cell, qp) = volPlasticStrainold(cell, qp) + devolps;

      } //loop over qps

    } //loop over cell

  } // end of evaluateFields

//**********************************************************************
// all local functions
  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT
  CapExplicit<EvalT, Traits>::compute_f(
      Intrepid2::Tensor<ScalarT> & sigma, Intrepid2::Tensor<ScalarT> & alpha,
      ScalarT & kappa)
  {

    xi = sigma - alpha;

    ScalarT I1 = Intrepid2::trace(xi), p = I1 / 3;

    s = xi - p * Intrepid2::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * Intrepid2::dotdot(s, s);

    ScalarT J3 = Intrepid2::det(s);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma = 0.5
          * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
              + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
  }

  template<typename EvalT, typename Traits>
  Intrepid2::Tensor<typename CapExplicit<EvalT, Traits>::ScalarT>
  CapExplicit<EvalT, Traits>::compute_dfdsigma(
    Intrepid2::Tensor<ScalarT> & sigma,
    Intrepid2::Tensor<ScalarT> & alpha, ScalarT & kappa)
  {

    xi = sigma - alpha;

    ScalarT I1 = Intrepid2::trace(xi), p = I1 / 3;

    s = xi - p * Intrepid2::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * Intrepid2::dotdot(s, s);

    ScalarT J3 = Intrepid2::det(s);

    //dI1dsigma = I;
    //dJ2dsigma = s;
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
      Gamma = 0.5
          * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
              + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    // derivatives
    ScalarT dFfdI1 = -(B * C * std::exp(B * I1) + theta);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0)
      dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0
          * (1.0 - 1.0 / psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0)
      dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0
          * (1.0 - 1.0 / psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    dfdsigma = dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;

    return dfdsigma;
  }

  template<typename EvalT, typename Traits>
  Intrepid2::Tensor<typename CapExplicit<EvalT, Traits>::ScalarT>
  CapExplicit<EvalT, Traits>::compute_dgdsigma(
    Intrepid2::Tensor<ScalarT> & sigma,
    Intrepid2::Tensor<ScalarT> & alpha, ScalarT & kappa)
  {

    xi = sigma - alpha;

    ScalarT I1 = Intrepid2::trace(xi), p = I1 / 3;

    s = xi - p * Intrepid2::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * Intrepid2::dotdot(s, s);

    ScalarT J3 = Intrepid2::det(s);

    //dJ2dsigma = s;
    //dJ3dsigma(3);
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
      Gamma = 0.5
          * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
              + (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

    // derivatives
    ScalarT dFfdI1 = -(L * C * std::exp(L * I1) + phi);

    ScalarT dFcdI1 = 0.0;
    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

    ScalarT dfdI1 = -(Ff_I1 - N) * (2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

    ScalarT dGammadJ2 = 0.0;
    if (J2 != 0)
      dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0
          * (1.0 - 1.0 / psi);

    ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

    ScalarT dGammadJ3 = 0.0;
    if (J2 != 0)
      dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0
          * (1.0 - 1.0 / psi);

    ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

    dgdsigma = dfdI1 * I + dfdJ2 * s + dfdJ3 * dJ3dsigma;

    return dgdsigma;
  }

  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT
  CapExplicit<EvalT, Traits>::compute_dfdkappa(
    Intrepid2::Tensor<ScalarT> & sigma, Intrepid2::Tensor<ScalarT> & alpha,
      ScalarT & kappa)
  {
    ScalarT dfdkappa;

    xi = sigma - alpha;

    ScalarT I1 = Intrepid2::trace(xi), p = I1 / 3.0;

    s = xi - p * Intrepid2::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * Intrepid2::dotdot(s, s);

    ScalarT J3 = Intrepid2::det(s);

    ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT dFcdkappa = 0.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0)) {
      dFcdkappa = 2 * (I1 - kappa)
          * ((X - kappa)
              + R * (I1 - kappa) * (theta + B * C * std::exp(B * kappa)))
          / (X - kappa) / (X - kappa) / (X - kappa);
    }

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
  Intrepid2::Tensor<typename CapExplicit<EvalT, Traits>::ScalarT>
  CapExplicit<EvalT, Traits>::compute_halpha(Intrepid2::Tensor<ScalarT> & dgdsigma,
      ScalarT & J2_alpha)
  {

    ScalarT Galpha = compute_Galpha(J2_alpha);

    ScalarT I1 = Intrepid2::trace(dgdsigma), p = I1 / 3;

    s = dgdsigma - p * I;

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        halpha(i, j) = calpha * Galpha * s(i, j);
      }
    }

    return halpha;
  }

  template<typename EvalT, typename Traits>
  typename CapExplicit<EvalT, Traits>::ScalarT
  CapExplicit<EvalT, Traits>::compute_dedkappa(
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
