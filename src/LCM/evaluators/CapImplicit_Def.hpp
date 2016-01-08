//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "LocalNonlinearSolver.hpp"
namespace LCM
{

//**********************************************************************
  template<typename EvalT, typename Traits>
  CapImplicit<EvalT, Traits>::
  CapImplicit(const Teuchos::ParameterList& p,
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

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void CapImplicit<EvalT, Traits>::postRegistrationSetup(
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
  void CapImplicit<EvalT, Traits>::evaluateFields(
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

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // local parameters
        ScalarT lame = elasticModulus(cell, qp) * poissonsRatio(cell, qp)
            / (1.0 + poissonsRatio(cell, qp))
            / (1.0 - 2.0 * poissonsRatio(cell, qp));
        ScalarT mu = elasticModulus(cell, qp) / 2.0
            / (1.0 + poissonsRatio(cell, qp));
        ScalarT bulkModulus = lame + (2. / 3.) * mu;

        // elastic matrix
        Intrepid2::Tensor4<ScalarT> Celastic = lame
            * Intrepid2::identity_3<ScalarT>(3)
            + mu
                * (Intrepid2::identity_1<ScalarT>(3)
                    + Intrepid2::identity_2<ScalarT>(3));

        // elastic compliance tangent matrix
        Intrepid2::Tensor4<ScalarT> compliance = (1. / bulkModulus / 9.)
            * Intrepid2::identity_3<ScalarT>(3)
            + (1. / mu / 2.)
                * (0.5
                    * (Intrepid2::identity_1<ScalarT>(3)
                        + Intrepid2::identity_2<ScalarT>(3))
                    - (1. / 3.) * Intrepid2::identity_3<ScalarT>(3));

        // previous state
        Intrepid2::Tensor<ScalarT>
        sigmaN(3, Intrepid2::ZEROS),
        alphaN(3, Intrepid2::ZEROS),
        strainN(3, Intrepid2::ZEROS);

        // incremental strain tensor
        Intrepid2::Tensor<ScalarT> depsilon(3);
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);
            strainN(i, j) = strainold(cell, qp, i, j);
          }
        }

        // trial state
        Intrepid2::Tensor<ScalarT> sigmaVal = Intrepid2::dotdot(Celastic,
            depsilon);
        Intrepid2::Tensor<ScalarT> alphaVal(3, Intrepid2::ZEROS);

        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            sigmaVal(i, j) = sigmaVal(i, j) + stressold(cell, qp, i, j);
            alphaVal(i, j) = backStressold(cell, qp, i, j);
            sigmaN(i, j) = stressold(cell, qp, i, j);
            alphaN(i, j) = backStressold(cell, qp, i, j);
          }
        }

        ScalarT kappaVal = capParameterold(cell, qp);
        ScalarT dgammaVal = 0.0;

        // used in defining generalized hardening modulus
        ScalarT Htan(0.0);

        // define plastic strain increment, its two invariants: dev, and vol
        Intrepid2::Tensor<ScalarT> deps_plastic(3, Intrepid2::ZEROS);
        ScalarT deqps(0.0), devolps(0.0);

        // define temporary trial stress, used in computing plastic strain
        Intrepid2::Tensor<ScalarT> sigmaTr = sigmaVal;

        std::vector<ScalarT> XXVal(13);

        // check yielding
        ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);
        XXVal = initialize(sigmaVal, alphaVal, kappaVal, dgammaVal);

        // local Newton loop
        if (f > 1.e-11) { // plastic yielding

          ScalarT normR, normR0, conv;
          bool kappa_flag = false;
          bool converged = false;
          int iter = 0;

          std::vector<ScalarT> R(13);
          std::vector<ScalarT> dRdX(13 * 13);
          LocalNonlinearSolver<EvalT, Traits> solver;

          while (!converged) {

            // assemble residual vector and local Jacobian
            compute_ResidJacobian(XXVal, R, dRdX, sigmaVal, alphaVal, kappaVal,
                Celastic, kappa_flag);

            normR = 0.0;
            for (int i = 0; i < 13; i++)
              normR += R[i] * R[i];

            normR = std::sqrt(normR);

            if (iter == 0)
              normR0 = normR;
            if (normR0 != 0)
              conv = normR / normR0;
            else
              conv = normR0;

            if (conv < 1.e-11 || normR < 1.e-11)
              break;

            if(iter > 20)
              break;

            //TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
             // std::endl << "Error in return mapping, iter = " << iter << "\nres = " << normR << "\nrelres = " << conv << std::endl);

            std::vector<ScalarT> XXValK = XXVal;
            solver.solve(dRdX, XXValK, R);

            // put restrictions on kappa: only allows monotonic decreasing (cap hardening)
            if (XXValK[11] > XXVal[11]) {
              kappa_flag = true;
            }
            else {
              XXVal = XXValK;
              kappa_flag = false;
            }

            // debugging
            //XXVal = XXValK;

            iter++;
          } //end local NR

          // compute sensitivity information, and pack back to X.
          solver.computeFadInfo(dRdX, XXVal, R);

        } // end of plasticity

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

        //dgammaVal = XXVal[12];

        //compute plastic strain increment deps_plastic = compliance ( sigma_tr - sigma_(n+1));
        Intrepid2::Tensor<ScalarT> dsigma = sigmaTr - sigmaVal;
        deps_plastic = Intrepid2::dotdot(compliance, dsigma);

        // compute its two invariants: devolps (volumetric) and deqps (deviatoric)
        devolps = Intrepid2::trace(deps_plastic);
        Intrepid2::Tensor<ScalarT> dev_plastic = deps_plastic
            - (1.0 / 3.0) * devolps * Intrepid2::identity<ScalarT>(3);
        //deqps = std::sqrt(2./3.) * Intrepid2::norm(dev_plastic);
        // use altenative definition, just differ by constants
        deqps = std::sqrt(2) * Intrepid2::norm(dev_plastic);

        //
        // update
        //
        // dilatancy
        //if (deqps != 0)
          //dilatancy(cell, qp) = devolps / deqps;
        //else
          //dilatancy(cell, qp) = 0.0;

        // friction coefficient = dtau / dp;
        // previous p and tau
        //ScalarT pN(0.0), tauN(0.0);
        //Intrepid2::Tensor<ScalarT> xi = sigmaN - alphaN;
        //pN = Intrepid2::trace(xi);
        //pN = pN / 3.0;
        //Intrepid2::Tensor<ScalarT> sN =
            //xi - pN * Intrepid2::identity<ScalarT>(3.0);
        //qN = sqrt(3./2.) * Intrepid2::norm(sN);
        //tauN = sqrt(1.0 / 2.0) * Intrepid2::norm(sN);

        // current p, and tau
        //ScalarT p(0.0), tau(0.0);
        //xi = sigmaVal - alphaVal;
        //p = Intrepid2::trace(xi);
        //p = p / 3.0;
        //Intrepid2::Tensor<ScalarT> s = xi - p * Intrepid2::identity<ScalarT>(3);
        //q = sqrt(3./2.) * Intrepid2::norm(s);
        //tau = sqrt(1.0 / 2.0) * Intrepid2::norm(s);
        //Intrepid2::Tensor<ScalarT, 3> ds = s - sN;

        // difference
        //ScalarT dtau = tau - tauN;
        //ScalarT dtau = sqrt(1./2.) * Intrepid2::norm(ds);
        //ScalarT dp = p - pN;

        // friction coefficient by finite difference
        //if (dp != 0)
          //friction(cell, qp) = dtau / dp;
        //else
          //friction(cell, qp) = 0.0;

        // hardening modulus
        // previous r(gamma)
        //ScalarT rN(0.0);
        //ScalarT evol3 = Intrepid2::trace(strainN);
        //evol3 = evol3 / 3.;
        //Intrepid2::Tensor<ScalarT> e = strainN
            //- evol3 * Intrepid2::identity<ScalarT>(3);
        //rN = sqrt(2.) * Intrepid2::norm(e);

        // current r(gamma)
        //ScalarT r(0.0);
        //Intrepid2::Tensor<ScalarT> strainCurrent = strainN + depsilon;
        //evol3 = Intrepid2::trace(strainCurrent);
        //evol3 = evol3 / 3.;
        //e = strainCurrent - evol3 * Intrepid2::identity<ScalarT>(3);
        //r = sqrt(2.) * Intrepid2::norm(e);

        // difference
        //ScalarT dr = r - rN;
        // tagent hardening modulus
        //if (dr != 0)
          //Htan = dtau / dr;

        //if (std::abs(1. - Htan / mu) > 0)
          //hardeningModulus(cell, qp) = Htan / (1. - Htan / mu);
        //else
          //hardeningModulus(cell, qp) = 0.0;

        // stress and back stress
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
  typename CapImplicit<EvalT, Traits>::ScalarT CapImplicit<EvalT, Traits>::compute_f(
      Intrepid2::Tensor<ScalarT> & sigma, Intrepid2::Tensor<ScalarT> & alpha,
      ScalarT & kappa)
  {

    Intrepid2::Tensor<ScalarT> xi = sigma - alpha;

    ScalarT I1 = Intrepid2::trace(xi), p = I1 / 3.;

    Intrepid2::Tensor<ScalarT> s = xi - p * Intrepid2::identity<ScalarT>(3);

    ScalarT J2 = 0.5 * Intrepid2::dotdot(s, s);

    ScalarT J3 = Intrepid2::det(s);

    ScalarT Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma =
          0.5
              * (1. - 3. * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
                  + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5))
                      / psi);

    ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    ScalarT X = kappa - R * Ff_kappa;

    ScalarT Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
  }

  template<typename EvalT, typename Traits>
  typename CapImplicit<EvalT, Traits>::DFadType CapImplicit<EvalT, Traits>::compute_f(
      Intrepid2::Tensor<DFadType> & sigma, Intrepid2::Tensor<DFadType> & alpha,
      DFadType & kappa)
  {

    Intrepid2::Tensor<DFadType> xi = sigma - alpha;

    DFadType I1 = Intrepid2::trace(xi), p = I1 / 3.;

    Intrepid2::Tensor<DFadType> s = xi - p * Intrepid2::identity<DFadType>(3);

    DFadType J2 = 0.5 * Intrepid2::dotdot(s, s);

    DFadType J3 = Intrepid2::det(s);

    DFadType Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma =
          0.5
              * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
                  + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5))
                      / psi);

    DFadType Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

    DFadType Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

    DFadType X = kappa - R * Ff_kappa;

    DFadType Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
  }

  template<typename EvalT, typename Traits>
  typename CapImplicit<EvalT, Traits>::D2FadType CapImplicit<EvalT, Traits>::compute_g(
      Intrepid2::Tensor<D2FadType> & sigma, Intrepid2::Tensor<D2FadType> & alpha,
      D2FadType & kappa)
  {

    Intrepid2::Tensor<D2FadType> xi = sigma - alpha;

    D2FadType I1 = Intrepid2::trace(xi), p = I1 / 3.;

    Intrepid2::Tensor<D2FadType> s = xi - p * Intrepid2::identity<D2FadType>(3);

    D2FadType J2 = 0.5 * Intrepid2::dotdot(s, s);

        D2FadType J3 = Intrepid2::det(s);

    D2FadType Gamma = 1.0;
    if (psi != 0 && J2 != 0)
      Gamma =
          0.5
              * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
                  + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5))
                      / psi);

    D2FadType Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

    D2FadType Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

    D2FadType X = kappa - Q * Ff_kappa;

    D2FadType Fc = 1.0;

    if ((kappa - I1) > 0 && ((X - kappa) != 0))
      Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

    return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
  }

  template<typename EvalT, typename Traits>
  std::vector<typename CapImplicit<EvalT, Traits>::ScalarT> CapImplicit<EvalT,
      Traits>::initialize(Intrepid2::Tensor<ScalarT> & sigmaVal,
      Intrepid2::Tensor<ScalarT> & alphaVal, ScalarT & kappaVal,
      ScalarT & dgammaVal)
  {
    std::vector<ScalarT> XX(13);

    XX[0] = sigmaVal(0, 0);
    XX[1] = sigmaVal(1, 1);
    XX[2] = sigmaVal(2, 2);
    XX[3] = sigmaVal(1, 2);
    XX[4] = sigmaVal(0, 2);
    XX[5] = sigmaVal(0, 1);
    XX[6] = alphaVal(0, 0);
    XX[7] = alphaVal(1, 1);
    XX[8] = alphaVal(1, 2);
    XX[9] = alphaVal(0, 2);
    XX[10] = alphaVal(0, 1);
    XX[11] = kappaVal;
    XX[12] = dgammaVal;

    return XX;
  }

  template<typename EvalT, typename Traits>
  Intrepid2::Tensor<typename CapImplicit<EvalT, Traits>::DFadType> CapImplicit<
      EvalT, Traits>::compute_dgdsigma(std::vector<DFadType> const & XX)
  {
    std::vector<D2FadType> D2XX(13);

    for (int i = 0; i < 13; ++i) {
      D2XX[i] = D2FadType(13, i, XX[i]);
    }

    Intrepid2::Tensor<D2FadType> sigma(3), alpha(3);

    sigma(0, 0) = D2XX[0];
    sigma(0, 1) = D2XX[5];
    sigma(0, 2) = D2XX[4];
    sigma(1, 0) = D2XX[5];
    sigma(1, 1) = D2XX[1];
    sigma(1, 2) = D2XX[3];
    sigma(2, 0) = D2XX[4];
    sigma(2, 1) = D2XX[3];
    sigma(2, 2) = D2XX[2];

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

    Intrepid2::Tensor<DFadType> dgdsigma(3);

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

  template<typename EvalT, typename Traits>
  typename CapImplicit<EvalT, Traits>::DFadType CapImplicit<EvalT, Traits>::compute_Galpha(
      DFadType J2_alpha)
  {
    if (N != 0)
      return 1.0 - pow(J2_alpha, 0.5) / N;
    else
      return 0.0;
  }

  template<typename EvalT, typename Traits>
  Intrepid2::Tensor<typename CapImplicit<EvalT, Traits>::DFadType> CapImplicit<
      EvalT, Traits>::compute_halpha(
      Intrepid2::Tensor<DFadType> const & dgdsigma, DFadType const J2_alpha)
  {

    DFadType Galpha = compute_Galpha(J2_alpha);

    DFadType I1 = Intrepid2::trace(dgdsigma), p = I1 / 3.0;

    Intrepid2::Tensor<DFadType> s = dgdsigma
        - p * Intrepid2::identity<DFadType>(3);

    //Intrepid2::Tensor<DFadType, 3> halpha = calpha * Galpha * s; // * operator not defined;
    Intrepid2::Tensor<DFadType> halpha(3);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        halpha(i, j) = calpha * Galpha * s(i, j);
      }
    }

    return halpha;
  }

  template<typename EvalT, typename Traits>
  typename CapImplicit<EvalT, Traits>::DFadType CapImplicit<EvalT, Traits>::compute_dedkappa(
      DFadType const kappa)
  {

    //******** use analytical expression
    ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;

    ScalarT X0 = kappa0 - Q * Ff_kappa0;

    DFadType Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

    DFadType X = kappa - Q * Ff_kappa;

    DFadType dedX = (D1 - 2. * D2 * (X - X0))
        * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

    DFadType dXdkappa = 1. + Q * C * L * exp(L * kappa) + Q * phi;

    return dedX * dXdkappa;
  }

  template<typename EvalT, typename Traits>
  typename CapImplicit<EvalT, Traits>::DFadType CapImplicit<EvalT, Traits>::compute_hkappa(
      DFadType const I1_dgdsigma, DFadType const dedkappa)
  {
    if (dedkappa != 0)
      return I1_dgdsigma / dedkappa;
    else
      return 0;
  }

  template<typename EvalT, typename Traits>
  void CapImplicit<EvalT, Traits>::compute_ResidJacobian(
      std::vector<ScalarT> const & XXVal, std::vector<ScalarT> & R,
      std::vector<ScalarT> & dRdX, const Intrepid2::Tensor<ScalarT> & sigmaVal,
      const Intrepid2::Tensor<ScalarT> & alphaVal, const ScalarT & kappaVal,
      Intrepid2::Tensor4<ScalarT> const & Celastic, bool kappa_flag)
  {

    std::vector<DFadType> Rfad(13);
    std::vector<DFadType> XX(13);
    std::vector<ScalarT> XXtmp(13);

    // initialize DFadType local unknown vector Xfad
    // Note that since Xfad is a temporary variable that gets changed within local iterations
    // when we initialize Xfad, we only pass in the values of X, NOT the system sensitivity information
    for (int i = 0; i < 13; ++i) {
      XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XXVal[i]);
      XX[i] = DFadType(13, i, XXtmp[i]);
    }

    Intrepid2::Tensor<DFadType> sigma(3), alpha(3);

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

    Intrepid2::Tensor<DFadType> dgdsigma = compute_dgdsigma(XX);

    DFadType J2_alpha = 0.5 * Intrepid2::dotdot(alpha, alpha);

    Intrepid2::Tensor<DFadType> halpha = compute_halpha(dgdsigma, J2_alpha);

    DFadType I1_dgdsigma = Intrepid2::trace(dgdsigma);

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

    // debugging
//	if(kappa_flag == false)Rfad[11] = -dgamma * hkappa - kappa + kappaVal;
//	else Rfad[11] = 0;

    Rfad[12] = f;

    // get ScalarT Residual
    for (int i = 0; i < 13; i++)
      R[i] = Rfad[i].val();

    //std::cout << "in assemble_Resid, R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;

    // get Jacobian
    for (int i = 0; i < 13; i++)
      for (int j = 0; j < 13; j++)
        dRdX[i + 13 * j] = Rfad[i].dx(j);

    if (kappa_flag == true) {
      for (int j = 0; j < 13; j++)
        dRdX[11 + 13 * j] = 0.0;

      dRdX[11 + 13 * 11] = 1.0;
    }

  }

} // end LCM
