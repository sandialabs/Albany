/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
 *                                                                    *
 * Notice: This computer software was prepared by Sandia Corporation, *
 * hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
 * the Department of Energy (DOE). All rights in the computer software*
 * are reserved by DOE on behalf of the United States Government and  *
 * the Contractor as provided in the Contract. You are authorized to  *
 * use this computer software for Governmental purposes but it is not *
 * to be released or distributed to the public. NEITHER THE GOVERNMENT*
 * NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
 * including this sentence must appear on any copies of this software.*
 *    Questions to Andy Salinger, agsalin@sandia.gov                  *
 \********************************************************************/

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  GursonSDStress<EvalT, Traits>::GursonSDStress(const Teuchos::ParameterList& p) :
      elasticModulus(p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), poissonsRatio(
          p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), strain(
          p.get<std::string>("Strain Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), voidVolume(
          p.get<std::string>("Void Volume Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), ep(
          p.get<std::string>("ep Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), yieldStrength(
          p.get<std::string>("Yield Strength Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), f0(
          p.get<double>("f0 Name")), Y0(p.get<double>("Y0 Name")), kw(
          p.get<double>("kw Name")), N(p.get<double>("N Name")), q1(
          p.get<double>("q1 Name")), q2(p.get<double>("q2 Name")), q3(
          p.get<double>("q3 Name")), eN(p.get<double>("eN Name")), sN(
          p.get<double>("sN Name")), fN(p.get<double>("fN Name")), fc(
          p.get<double>("fc Name")), ff(p.get<double>("ff Name")), flag(
          p.get<double>("flag Name"))
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
    voidVolumeName = p.get<std::string>("Void Volume Name") + "_old";
    epName = p.get<std::string>("ep Name") + "_old";
    yieldStrengthName = p.get<std::string>("Yield Strength Name") + "_old";

    // evaluated fields
    this->addEvaluatedField(stress);
    this->addEvaluatedField(voidVolume);
    this->addEvaluatedField(ep);
    this->addEvaluatedField(yieldStrength);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void GursonSDStress<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(elasticModulus, fm);
    if (numDims > 1) this->utils.setFieldData(poissonsRatio, fm);
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(strain, fm);
    this->utils.setFieldData(voidVolume, fm);
    this->utils.setFieldData(ep, fm);
    this->utils.setFieldData(yieldStrength, fm);
  }

//**********************************************************************

  template<typename EvalT, typename Traits>
  void GursonSDStress<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    // previous state
    Albany::MDArray strainold = (*workset.stateArrayPtr)[strainName];
    Albany::MDArray stressold = (*workset.stateArrayPtr)[stressName];
    Albany::MDArray voidVolumeold = (*workset.stateArrayPtr)[voidVolumeName];
    Albany::MDArray epold = (*workset.stateArrayPtr)[epName];
    Albany::MDArray yieldStrengthold =
        (*workset.stateArrayPtr)[yieldStrengthName];

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // local parameters
        ScalarT lame = elasticModulus(cell, qp) * poissonsRatio(cell, qp)
            / (1.0 + poissonsRatio(cell, qp))
            / (1.0 - 2.0 * poissonsRatio(cell, qp));
        ScalarT mu = elasticModulus(cell, qp) / 2.0
            / (1.0 + poissonsRatio(cell, qp));

        ScalarT Eor3mu = elasticModulus(cell, qp);
        // if flag != 1, then use 3mu
        if (std::abs(flag - 1) > 1.0e-10) Eor3mu = 3. * mu;

        // elastic matrix
        LCM::Tensor4<ScalarT> Celastic = lame * LCM::identity_3<ScalarT>(3)
            + mu * (LCM::identity_1<ScalarT>(3) + LCM::identity_2<ScalarT>(3));

        // incremental strain tensor
        LCM::Tensor<ScalarT> depsilon(3, 0.0), stressN(3, 0.0);
        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);
            stressN(i, j) = stressold(cell, qp, i, j);
          }
        }

// 		// previous state
        ScalarT pN = (1. / 3.) * LCM::trace(stressN);
        LCM::Tensor<ScalarT> devN = stressN - pN * LCM::identity<ScalarT>(3);
        ScalarT sigeN = std::sqrt(LCM::dotdot(devN, devN)) * std::sqrt(3. / 2.);
        ScalarT J3N = LCM::det(devN);
        ScalarT fvoidN = voidVolumeold(cell, qp);
        ScalarT epN = epold(cell, qp);
        ScalarT YN = yieldStrengthold(cell, qp);
// 
// 		// trial state
        LCM::Tensor<ScalarT> stressVal = stressN
            + LCM::dotdot(Celastic, depsilon);
        ScalarT pVal = (1. / 3.) * LCM::trace(stressVal);
        LCM::Tensor<ScalarT> devVal = stressVal
            - pVal * LCM::identity<ScalarT>(3);
        ScalarT sigeVal = std::sqrt(LCM::dotdot(devVal, devVal))
            * std::sqrt(3. / 2.);
        ScalarT J3Val = LCM::det(devVal);

        ScalarT fvoidVal = fvoidN;
        ScalarT epVal = epN;
        ScalarT YVal = YN;
        ScalarT dgam = 0.0;

// 		// yield function
        ScalarT Phi = compute_Phi(devVal, pVal, fvoidVal, epVal, Eor3mu);

        if (Phi > 1.0e-10) {
          ScalarT tmp = 1.5 * q2 * pN / YN;
          ScalarT tmpfac = q1 * q2 * (1. / 3.) * YN * fvoidN * std::sinh(tmp);
          LCM::Tensor<ScalarT> dPhidsigma = devN
              + tmpfac * LCM::identity<ScalarT>(3);

          ScalarT dPhidf = -(2. / 3.) * YN * YN
              * (q3 * fvoidN - q1 * std::cosh(tmp));

          ScalarT dPhidep = compute_dPhidep(tmp, pN, fvoidN, epN, YN, Eor3mu);

          ScalarT depdgam;
          if (sigeN != 0)
            depdgam = (2. / 3. * sigeN * sigeN
                + q1 * q2 * pN * YN * fvoidN * std::sinh(tmp)) / (1 - fvoidN)
                / YN;
          else
            depdgam = (q1 * q2 * pN * YN * fvoidN * std::sinh(tmp))
                / (1 - fvoidN) / YN;

// 			// dfdgam;
          ScalarT dfdgam = compute_dfdgam(depdgam, tmp, pN, sigeN, J3N, fvoidN,
              epN, YN);

          ScalarT kai(0.0);
          kai = LCM::dotdot(dPhidsigma, LCM::dotdot(Celastic, dPhidsigma));
          kai = kai - dPhidf * dfdgam - dPhidep * depdgam;

          LCM::Tensor<ScalarT> dPhidotCe = LCM::dotdot(dPhidsigma, Celastic);

          // incremental consistency parameter
          if (kai != 0)
            dgam = LCM::dotdot(dPhidotCe, depsilon) / kai;
          else
            dgam = 0;

          //update
          stressVal -= dgam * LCM::dotdot(Celastic, dPhidsigma);
          fvoidVal += dgam * dfdgam;
          epVal += dgam * depdgam;
          YVal = compute_Y(epVal, Eor3mu);

          bool converged = false;
          int iter = 0;
          while (!converged) {
            pVal = (1. / 3.) * LCM::trace(stressVal);
            devVal = stressVal - pVal * LCM::identity<ScalarT>(3);
            sigeVal = std::sqrt(LCM::dotdot(devVal, devVal))
                * std::sqrt(3. / 2.);
            J3Val = LCM::det(devVal);

            Phi = compute_Phi(devVal, pVal, fvoidVal, epVal, Eor3mu);
            tmp = 1.5 * q2 * pVal / YVal;
            tmpfac = q1 * q2 * (1. / 3.) * YVal * fvoidVal * std::sinh(tmp);

            dPhidsigma = devVal + tmpfac * LCM::identity<ScalarT>(3);

            dPhidf = -(2. / 3.) * YVal * YVal
                * (q3 * fvoidVal - q1 * std::cosh(tmp));

            dPhidep = compute_dPhidep(tmp, pVal, fvoidVal, epVal, YVal, Eor3mu);

            if (sigeVal != 0)
              depdgam = (2. / 3. * sigeVal * sigeVal
                  + q1 * q2 * pVal * YVal * fvoidVal * std::sinh(tmp))
                  / (1 - fvoidVal) / YVal;
            else
              depdgam = (q1 * q2 * pVal * YVal * fvoidVal * std::sinh(tmp))
                  / (1 - fvoidVal) / YVal;

            dfdgam = compute_dfdgam(depdgam, tmp, pVal, sigeVal, J3Val,
                fvoidVal, epVal, YVal);

            kai = LCM::dotdot(dPhidsigma, LCM::dotdot(Celastic, dPhidsigma));
            kai = kai - dPhidf * dfdgam - dPhidep * depdgam;

            if ((std::abs(Phi) < 1.0e-10) || (iter > 12)) break;

            ScalarT delta_gam;
            if (kai != 0)
              delta_gam = Phi / kai;
            else
              delta_gam = 0;

            LCM::Tensor<ScalarT> stressK(3);
            ScalarT fvoidK, epK, YK;

            stressK = stressVal - delta_gam * LCM::dotdot(Celastic, dPhidsigma);
            fvoidK = fvoidVal + delta_gam * dfdgam;
            epK = epVal + delta_gam * depdgam;
            YK = compute_Y(epK, Eor3mu);

            ScalarT pK = (1. / 3.) * LCM::trace(stressK);
            LCM::Tensor<ScalarT> devK = stressK
                - pK * LCM::identity<ScalarT>(3);

            ScalarT Phipre = compute_Phi(devK, pK, fvoidK, epK, Eor3mu);

            if (std::abs(Phipre) > std::abs(Phi)) {
              // if the corrected stress is further away from yield surface, then use normal correction
              delta_gam = Phi / LCM::dotdot(dPhidsigma, dPhidsigma);
              stressK = stressVal - delta_gam * dPhidsigma;
              fvoidK = fvoidVal;
              epK = epVal;
              YK = compute_Y(epK, Eor3mu);
            }

            stressVal = stressK;
            fvoidVal = fvoidK;
            epVal = epK;
            YVal = YK;
            iter++;
          } // end of stress correction

        } // end plasticity

        // update
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            stress(cell, qp, i, j) = stressVal(i, j);

        voidVolume(cell, qp) = fvoidVal;
        ep(cell, qp) = epVal;
        yieldStrength(cell, qp) = compute_Y(epVal, Eor3mu);

      } //loop over qps
    } //loop over cell

  } // end of evaluateFields

//**********************************************************************
// all local functions
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT GursonSDStress<EvalT, Traits>::compute_Y(
      ScalarT & epVal, ScalarT & Eor3mu)
  {
    //ScalarT Eor3mu = E;  // fac = E in Nahshon, fac = 3*mu in Aravas;
    // a local NY to solve for Y
    int iter = 0;
    bool converged = false;
    ScalarT R, dRdY, tmp, Yk(Y0);

    while (!converged) {
      tmp = Yk / Y0 + Eor3mu * epVal / Y0;
      R = Yk - Y0 * std::pow(tmp, N);
      if (std::abs(R) < 1.0e-8 || iter > 8) break;
      dRdY = 1 - N * std::pow(tmp, N - 1);
      Yk = Yk - (1. / dRdY) * R;
      iter++;
    }
    return Yk;
  }

  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT GursonSDStress<EvalT, Traits>::compute_Phi(
      LCM::Tensor<ScalarT> & devVal, ScalarT & pVal, ScalarT & fvoidVal,
      ScalarT & epVal, ScalarT & Eor3mu)
  {
    ScalarT Y = compute_Y(epVal, Eor3mu);

    ScalarT tmp = 1.5 * q2 * pVal / Y;

    ScalarT fstar;
//	ScalarT ffbar = (q1 + std::sqrt(q1 * q1 -  q3)) / q3;
//	if(fvoidVal <= fc)
//		fstar = fvoidVal;
//	else if((fvoidVal > fc) && (fvoidVal < ff))
//		fstar = fc + (ffbar - fc) * (fvoidVal - fc) / (ff - fc);
//	else
//		fstar = ffbar;

    // As in Aravas 1987, and simple shearing test in Nahshon and Xue 2009
    fstar = fvoidVal;

    ScalarT psi = 1 + q3 * fstar * fstar - 2. * q1 * fstar * std::cosh(tmp);

    return 0.5 * LCM::dotdot(devVal, devVal) - psi * Y * Y / 3.0;
  }

  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT GursonSDStress<EvalT, Traits>::compute_dPhidep(
      ScalarT & tmp, ScalarT & pN, ScalarT & fvoidN, ScalarT & epN,
      ScalarT & YN, ScalarT & Eor3mu)
  {
    ScalarT psi = 1. + q3 * fvoidN * fvoidN - 2. * q1 * fvoidN * std::cosh(tmp);
    // note: corrected the sign typo in Steinmann et al eq(55)
    ScalarT dPhidY = (2. / 3.) * psi * YN
        - q1 * q2 * fvoidN * pN * std::sinh(tmp); // + , -
    ScalarT ratio = YN / Y0 + Eor3mu * epN / Y0;
    ScalarT dYdep = Eor3mu * N * std::pow(ratio, N - 1);

    return dPhidY * dYdep;
  }

  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT GursonSDStress<EvalT, Traits>::compute_dfdgam(
      ScalarT & depdgam, ScalarT & tmp, ScalarT & pN, ScalarT & sigeN,
      ScalarT & J3N, ScalarT & fvoidN, ScalarT & epN, ScalarT & YN)
  {

    ScalarT eratio = -0.5 * (epN - eN) * (epN - eN) / sN / sN;
    ScalarT A(0.0);
    const double pi = acos(-1.0);
    if (pN >= 0) A = fN / sN / (std::sqrt(2.0 * pi)) * std::exp(eratio);

    ScalarT dfndgam = A * depdgam;

    ScalarT omega(0.0);
    if (sigeN != 0) omega = 1
        - (27.0 * J3N / (2 * sigeN * sigeN * sigeN))
            * (27.0 * J3N / (2 * sigeN * sigeN * sigeN));

    ScalarT dfgdgam;
    if (sigeN != 0)
      dfgdgam = q1 * q2 * fvoidN * (1 - fvoidN) * YN * std::sinh(tmp)
          + 2. / 3. * kw * fvoidN * omega * sigeN;
    else
      dfgdgam = q1 * q2 * fvoidN * (1 - fvoidN) * YN * std::sinh(tmp);

    return dfgdgam + dfndgam;
  }
//**********************************************************************
}// end LCM
