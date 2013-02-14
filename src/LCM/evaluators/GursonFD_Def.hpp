//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "LocalNonlinearSolver.h"

namespace LCM
{

//**********************************************************************
  template<typename EvalT, typename Traits>
  GursonFD<EvalT, Traits>::GursonFD(const Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl) :
    deltaTime(p.get<std::string>("Delta Time Name"),dl->workset_scalar),
    defgrad(p.get<std::string>("DefGrad Name"),dl->qp_tensor),
    J(p.get<std::string>("DetDefGrad Name"),dl->qp_scalar),
    elasticModulus(p.get<std::string>("Elastic Modulus Name"),dl->qp_scalar),
    poissonsRatio(p.get<std::string>("Poissons Ratio Name"),dl->qp_scalar),
    yieldStrength(p.get<std::string>("Yield Strength Name"),dl->qp_scalar),
    hardeningModulus(p.get<std::string>("Hardening Modulus Name"),dl->qp_scalar),
    saturationModulus(p.get<std::string>("Saturation Modulus Name"),dl->qp_scalar),
    saturationExponent(p.get<std::string>("Saturation Exponent Name"),dl->qp_scalar),
    stress(p.get<std::string>("Stress Name"),dl->qp_tensor),
    Fp(p.get<std::string>("Fp Name"),dl->qp_tensor),
    eqps(p.get<std::string>("Eqps Name"),dl->qp_scalar),
    voidVolume(p.get<std::string>("Void Volume Name"),dl->qp_scalar)
  {
    // Pull out numQPs and numDims from a Layout
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(deltaTime);
    this->addDependentField(elasticModulus);
    // PoissonRatio not used in 1D stress calc
    if (numDims > 1)
      this->addDependentField(poissonsRatio);
    this->addDependentField(defgrad);
    this->addDependentField(J);
    this->addDependentField(yieldStrength);
    this->addDependentField(hardeningModulus);
    this->addDependentField(saturationModulus);
    this->addDependentField(saturationExponent);

    // state variable
    fpName = p.get<std::string>("Fp Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";
    voidVolumeName = p.get<std::string>("Void Volume Name") + "_old";
    defGradName = p.get<std::string>("DefGrad Name") + "_old";
    stressName = p.get<std::string>("Stress Name") + "_old";

    // evaluated fields
    this->addEvaluatedField(stress);
    this->addEvaluatedField(Fp);
    this->addEvaluatedField(eqps);
    this->addEvaluatedField(voidVolume);

    // scratch space FCs
    Fpinv.resize(worksetSize, numQPs, numDims, numDims);
    FpinvT.resize(worksetSize, numQPs, numDims, numDims);
    Cpinv.resize(worksetSize, numQPs, numDims, numDims);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);

    // get parameter list of material constants
    pList = p.get<Teuchos::ParameterList*>("Parameter List");

    // set material parameters

    N = pList->get<RealType>("N");
    eq0 = pList->get<RealType>("eq0");
    f0 = pList->get<RealType>("f0");
    kw = pList->get<RealType>("kw");
    eN = pList->get<RealType>("eN");
    sN = pList->get<RealType>("sN");
    fN = pList->get<RealType>("fN");
    fc = pList->get<RealType>("fc");
    ff = pList->get<RealType>("ff");
    q1 = pList->get<RealType>("q1");
    q2 = pList->get<RealType>("q2");
    q3 = pList->get<RealType>("q3");
    isSaturationH = pList->get<bool>("isSaturationH");

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void GursonFD<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(deltaTime, fm);
    this->utils.setFieldData(elasticModulus, fm);
    if (numDims > 1)
      this->utils.setFieldData(poissonsRatio, fm);
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(defgrad, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(yieldStrength, fm);
    this->utils.setFieldData(saturationModulus, fm);
    this->utils.setFieldData(saturationExponent, fm);
    this->utils.setFieldData(Fp, fm);
    this->utils.setFieldData(eqps, fm);
    this->utils.setFieldData(voidVolume, fm);
  }

//**********************************************************************

  template<typename EvalT, typename Traits>
  void GursonFD<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    typedef Intrepid::FunctionSpaceTools FST;
    typedef Intrepid::RealSpaceTools<ScalarT> RST;

    ScalarT sq23 = std::sqrt(2. / 3.);
    ScalarT sq32 = std::sqrt(3. / 2.);

    // previous state
    Albany::MDArray FpOld = (*workset.stateArrayPtr)[fpName];
    Albany::MDArray eqpsOld = (*workset.stateArrayPtr)[eqpsName];
    Albany::MDArray voidVolumeold = (*workset.stateArrayPtr)[voidVolumeName];
    Albany::MDArray defGradOld = (*workset.stateArrayPtr)[defGradName];
    Albany::MDArray stressOld = (*workset.stateArrayPtr)[stressName];

    // compute Cp_{n}^{-1}
    // AGS MAY NEED TO ALLICATE Fpinv FpinvT Cpinv  with actual workse size
    // to prevent going past the end of Fpold.
    RST::inverse(Fpinv, FpOld);
    RST::transpose(FpinvT, Fpinv);
    FST::tensorMultiplyDataData<ScalarT>(Cpinv, Fpinv, FpinvT);

    ScalarT bulkModulus, shearModulus, lame;
    ScalarT K, Y, siginf, delta;
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {

        bulkModulus = elasticModulus(cell, qp)
            / (3. * (1. - 2. * poissonsRatio(cell, qp)));
        shearModulus = elasticModulus(cell, qp)
            / (2. * (1. + poissonsRatio(cell, qp)));
        lame = bulkModulus - 2. * shearModulus / 3.;
        K = hardeningModulus(cell, qp);
        Y = yieldStrength(cell, qp);
        siginf = saturationModulus(cell, qp);
        delta = saturationExponent(cell, qp);

        //
        Tensor Fnew(defgrad(cell, qp, 0, 0), defgrad(cell, qp, 0, 1),
            defgrad(cell, qp, 0, 2), defgrad(cell, qp, 1, 0),
            defgrad(cell, qp, 1, 1), defgrad(cell, qp, 1, 2),
            defgrad(cell, qp, 2, 0), defgrad(cell, qp, 2, 1),
            defgrad(cell, qp, 2, 2));
        Tensor s(3);
        ScalarT p;

        // Compute Trial State
        // Hyperelastic
        Tensor CpinvOld(Cpinv(cell, qp, 0, 0), Cpinv(cell, qp, 0, 1),
            Cpinv(cell, qp, 0, 2), Cpinv(cell, qp, 1, 0),
            Cpinv(cell, qp, 1, 1), Cpinv(cell, qp, 1, 2),
            Cpinv(cell, qp, 2, 0), Cpinv(cell, qp, 2, 1),
            Cpinv(cell, qp, 2, 2));

        Tensor be = Intrepid::dot(Fnew,
              Intrepid::dot(CpinvOld, Intrepid::transpose(Fnew)));
        Tensor logbe = Intrepid::log<ScalarT>(be);
        ScalarT trlogbeby3 = Intrepid::trace(logbe) / 3.0;
        ScalarT detbe = Intrepid::det<ScalarT>(be);

        s = shearModulus
              * (logbe - trlogbeby3 * Intrepid::identity<ScalarT>(3));
        p = 0.5 * bulkModulus * std::log(detbe);

        ScalarT fvoid = voidVolumeold(cell, qp);
        ScalarT eq = eqpsOld(cell, qp);

        ScalarT Phi = YieldFunction(s, p, fvoid, eq, K, Y, siginf, delta,
            J(cell, qp), elasticModulus(cell, qp));

        ScalarT dgam(0.0);
        if (Phi > 0.0) { // plastic yielding

          // initialize local unknown vector
          std::vector<ScalarT> X(4);
          X[0] = dgam;
          X[1] = p;
          X[2] = fvoid;
          X[3] = eq;

          LocalNonlinearSolver<EvalT, Traits> solver;

          const int maxIter = 20;
          const RealType tolerance = 1.e-11;
          int iter = 0;
          ScalarT normR0(0.0), relativeR(0.0), normR(0.0);
          std::vector<ScalarT> R(4);
          std::vector<ScalarT> dRdX(16);

          // local N-R loop
          while (true) {
            ResidualJacobian(X, R, dRdX, p, fvoid, eq, s, shearModulus,
                bulkModulus, K, Y, siginf, delta, J(cell, qp));

            normR = 0.0;
            for (int i = 0; i < 4; i++)
              normR += R[i] * R[i];

            normR = std::sqrt(normR);

            if (iter == 0)
              normR0 = normR;
            if (normR0 != 0)
              relativeR = normR / normR0;
            else
              relativeR = normR0;

            //std::cout << iter << " " << normR << " " << relativeR << std::endl;
            if (relativeR < tolerance || normR < tolerance)
              break;
            if (iter > maxIter)
              break;

            //TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
            //std::endl << "Error in return mapping, iter = "
            //<< iter << "\nres = " << normR << "\nrelres = " << relativeR << std::endl);

            // call local nonlinear solver
            solver.solve(dRdX, X, R);

            iter++;
          } // end of local N-R

          // compute sensitivity information w.r.t system parameters,
          // and pack back to X
          solver.computeFadInfo(dRdX, X, R);

          // update
          dgam = X[0];
          p = X[1];
          fvoid = X[2];
          eq = X[3];

          // accounts for void coalescence
          ScalarT fvoidStar = fvoid;
          if ((fvoid > fc) && (fvoid < ff)) {
            if ((ff - fc) != 0.0) {
              fvoidStar = fc + (fvoid - fc) * (1. / q1 - fc) / (ff - fc);
            }
          }
          else if (fvoid >= ff) {
            fvoidStar = 1. / q1;
            if (fvoidStar > 1.0)
              fvoidStar = 1.0;
          }

          for (std::size_t i = 0; i < numDims; ++i)
            for (std::size_t j = 0; j < numDims; ++j)
              s(i, j) = (1. / (1. + 2. * shearModulus * dgam)) * s(i, j);

          // Yield strength
            ScalarT Ybar(0.0);

            if (isSaturationH == true) { // original saturation type hardening
              ScalarT h = siginf * (1. - std::exp(-delta * eq)) + K * eq;
              Ybar = Y + h;
            }
            else { // powerlaw hardening
              ScalarT x = 1. + elasticModulus(cell, qp) * eq / Y;
              //ScalarT x = eq0 + eq;
              Ybar = Y * std::pow(x, N);
            }

            Ybar = Ybar * J(cell, qp);

            ScalarT tmp = 1.5 * q2 * p / Ybar;

            Tensor dPhi(3, 0.0);

            for (std::size_t i = 0; i < numDims; ++i) {
              for (std::size_t j = 0; j < numDims; ++j) {
                dPhi(i, j) = s(i, j);
              }
              dPhi(i, i) += 1. / 3. * q1 * q2 * Ybar * fvoidStar
                  * std::sinh(tmp);
            }

            Tensor A = dgam * dPhi;
            Tensor expA = Intrepid::exp(A);

            for (std::size_t i = 0; i < numDims; ++i) {
              for (std::size_t j = 0; j < numDims; ++j) {
                Fp(cell, qp, i, j) = 0.0;
                for (std::size_t p = 0; p < numDims; ++p) {
                  Fp(cell, qp, i, j) += expA(i, p) * FpOld(cell, qp, p, j);
                }
              }
            }

          eqps(cell, qp) = eq;
          voidVolume(cell, qp) = fvoid;

        } // end of plastic loading
        else { // elasticity, set state variables to old values
          eqps(cell, qp) = eqpsOld(cell, qp);
          voidVolume(cell, qp) = voidVolumeold(cell, qp);

          for (std::size_t i = 0; i < numDims; ++i)
            for (std::size_t j = 0; j < numDims; ++j)
              Fp(cell, qp, i, j) = FpOld(cell, qp, i, j);

        } // end of elasticity

        // compute Cauchy stress tensor
        // (note that p also has to be divided by J, since its the Kirchhoff pressure)
        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
             stress(cell, qp, i, j) = s(i, j) / J(cell, qp);
          }
          stress(cell, qp, i, i) += p / J(cell, qp);
        }

      } // end of loop over qp
    } //end of loop over cell

    // Since Intrepid will later perform calculations on the entire workset size
    // and not just the used portion, we must fill the excess with reasonable
    // values. Leaving this out leads to inversion of 0 tensors.
      for (std::size_t cell = workset.numCells; cell < worksetSize; ++cell)
        for (std::size_t qp = 0; qp < numQPs; ++qp)
          for (std::size_t i = 0; i < numDims; ++i)
            Fp(cell, qp, i, i) = 1.0;

  } // end of evaluateFields

//**********************************************************************
// all local functions
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT GursonFD<EvalT, Traits>::YieldFunction(
      Tensor const & s, ScalarT const & p, ScalarT const & fvoid,
      ScalarT const & eq, ScalarT const & K, ScalarT const & Y,
      ScalarT const & siginf, ScalarT const & delta, ScalarT const & Jacobian,
      ScalarT const & E)
  {

    // Yield strength
    ScalarT Ybar(0.0);

    if (isSaturationH == true) { // original saturation type hardening
      ScalarT h = siginf * (1. - std::exp(-delta * eq)) + K * eq;
      Ybar = Y + h;
    }
    else { // powerlaw hardening
      ScalarT x = 1. + E * eq / Y;
      //ScalarT x = eq0 + eq;
      Ybar = Y * std::pow(x, N);
    }

    // Kirchhoff yield stress
    Ybar = Ybar * Jacobian;

    ScalarT tmp = 1.5 * q2 * p / Ybar;

    // accounts for void coalescence
    ScalarT fvoidStar = fvoid;
    if ((fvoid > fc) && (fvoid < ff)) {
      if ((ff - fc) != 0.0) {
        fvoidStar = fc + (fvoid - fc) * (1. / q1 - fc) / (ff - fc);
      }
    }
    else if (fvoid >= ff) {
      fvoidStar = 1. / q1;
      if (fvoidStar > 1.0)
        fvoidStar = 1.0;
    }

    ScalarT psi = 1. + q3 * fvoidStar * fvoidStar
        - 2. * q1 * fvoidStar * std::cosh(tmp);

    // a quadratic representation will look like:
    ScalarT Phi = 0.5 * Intrepid::dotdot(s, s) - psi * Ybar * Ybar / 3.0;

    // linear form
//	ScalarT smag = Intrepid::dotdot(s,s);
//	smag = std::sqrt(smag);
//	ScalarT sq23 = std::sqrt(2./3.);
//  ScalarT Phi = smag - sq23 * std::sqrt(psi) * psi_sign * Ybar

    return Phi;
  }

  template<typename EvalT, typename Traits>
  void GursonFD<EvalT, Traits>::ResidualJacobian(std::vector<ScalarT> & X,
      std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX, const ScalarT & p,
      const ScalarT & fvoid, const ScalarT & eq, Tensor & s,
      const ScalarT & shearModulus, const ScalarT & bulkModulus,
      const ScalarT & K, const ScalarT & Y, const ScalarT & siginf,
      const ScalarT & delta, const ScalarT & Jacobian)
  {
    ScalarT sq32 = std::sqrt(3. / 2.);
    ScalarT sq23 = std::sqrt(2. / 3.);
    std::vector<DFadType> Rfad(4);
    std::vector<DFadType> Xfad(4);

    // initialize DFadType local unknown vector Xfad
    // Note that since Xfad is a temporary variable
    // that gets changed within local iterations
    // when we initialize Xfad, we only pass in the values of X,
    // NOT the system sensitivity information
    std::vector<ScalarT> Xval(4);
    for (std::size_t i = 0; i < 4; ++i) {
      Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i] = DFadType(4, i, Xval[i]);
    }

    DFadType dgam = Xfad[0], pFad = Xfad[1], fvoidFad = Xfad[2],
        eqFad = Xfad[3];

    // accounts for void coalescence
    DFadType fvoidFadStar = fvoidFad;

    if ((fvoidFad > fc) && (fvoidFad < ff)) {
      if ((ff - fc) != 0.0) {
        fvoidFadStar = fc + (fvoidFad - fc) * (1. / q1 - fc) / (ff - fc);
      }
    }
    else if (fvoidFad >= ff) {
      fvoidFadStar = 1. / q1;
      if (fvoidFadStar > 1.0)
        fvoidFadStar = 1.0;
    }

    // have to break down these equations, otherwise I get compile error
    // Yield strength
    DFadType Ybar(0.0);

    if (isSaturationH) { // original saturation type hardening
      DFadType h(0.0); // h = siginf * (1. - std::exp(-delta*eqFad)) + K * eqFad;
      h = delta * eqFad;
      h = -1. * h;
      h = std::exp(h);
      h = 1. - h;
      h = siginf * h;
      h = h + K * eqFad;

      Ybar = Y + h;
    }
    else { // powerlaw hardening
      ScalarT E = 9. * bulkModulus * shearModulus
          / (3. * bulkModulus + shearModulus);
      DFadType x(0.0); // x = 1. + E * eqFad / Y;
      x = E * eqFad;
      x = x / Y;
      x = 1.0 + x;
      //DFadType x = eqFad + eq0;
      Ybar = Y * std::pow(x, N);
    }

    // Kirchhoff yield stress
    Ybar = Ybar * Jacobian;

    DFadType tmp = pFad / Ybar;
    tmp = 1.5 * tmp;
    tmp = q2 * tmp;

    DFadType fvoid2;
    fvoid2 = fvoidFadStar * fvoidFadStar;
    fvoid2 = q3 * fvoid2;

    DFadType psi;
    psi = std::cosh(tmp);
    psi = fvoidFadStar * psi;
    psi = 2. * psi;
    psi = q1 * psi;
    psi = fvoid2 - psi;
    psi = 1. + psi;

    DFadType fac; // fac = 1./(1. + (2.* (shearModulus * dgam)))
    fac = shearModulus * dgam;
    fac = 2. * fac;
    fac = 1. + fac;
    fac = 1. / fac;

    Intrepid::Tensor<DFadType> sfad(3, 0.0);

    // valid for assumption Ntr = N;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        sfad(i, j) = fac * s(i, j);
      }
    }
    // shear-dependent term in void growth
    DFadType omega(0.0), J3(0.0), taue(0.0), smag2, smag;
    J3 = Intrepid::det(sfad);
    smag2 = Intrepid::dotdot(sfad, sfad);
    if (smag2 > 0.0) {
      smag = std::sqrt(smag2);
      taue = sq32 * smag;
    }

    if (taue > 0.0)
      omega = 1.
          - (27. * J3 / 2. / taue / taue / taue)
              * (27. * J3 / 2. / taue / taue / taue);

    DFadType deq(0.0);
    if (smag != 0.0) {
      deq = dgam
          * (smag2 + q1 * q2 * pFad * Ybar * fvoidFadStar * std::sinh(tmp))
          / (1. - fvoidFad) / Ybar;
    }
    else {
      deq = dgam * (q1 * q2 * pFad * Ybar * fvoidFadStar * std::sinh(tmp))
          / (1. - fvoidFad) / Ybar;
    }
    // void nucleation
    DFadType dfn(0.0);
    DFadType An(0.0), eratio(0.0);
    eratio = -0.5 * (eqFad - eN) * (eqFad - eN) / sN / sN;

    const double pi = acos(-1.0);
    if (pFad >= 0.0) {
      An = fN / sN / (std::sqrt(2.0 * pi)) * std::exp(eratio);
    }

    dfn = An * deq;

    DFadType dfg(0.0);
    if (taue > 0.0) {
      dfg = dgam * q1 * q2 * (1. - fvoidFad) * fvoidFadStar * Ybar
          * std::sinh(tmp) + sq23 * dgam * kw * fvoidFad * omega * smag;
    }
    else {
      dfg = dgam * q1 * q2 * (1. - fvoidFad) * fvoidFad * Ybar * std::sinh(tmp);
    }

    DFadType Phi;

    Phi = 0.5 * smag2 - psi * Ybar * Ybar / 3.0;

    // local system of equations
    Rfad[0] = Phi;
    Rfad[1] = pFad - p
        + dgam * q1 * q2 * bulkModulus * Ybar * fvoidFad * std::sinh(tmp);
    Rfad[2] = fvoidFad - fvoid - dfg - dfn;
    Rfad[3] = eqFad - eq - deq;

    // get ScalarT Residual
    for (int i = 0; i < 4; i++)
      R[i] = Rfad[i].val();

    // get local Jacobian
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        dRdX[i + 4 * j] = Rfad[i].dx(j);

  }

//**********************************************************************
}// end LCM
