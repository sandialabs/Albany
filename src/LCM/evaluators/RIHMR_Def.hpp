//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "LocalNonlinearSolver.h"

#include <typeinfo>

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  RIHMR<EvalT, Traits>::RIHMR(const Teuchos::ParameterList& p) :
      defgrad(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), elasticModulus(
          p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), poissonsRatio(
          p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), yieldStrength(
          p.get<std::string>("Yield Strength Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), hardeningModulus(
          p.get<std::string>("Hardening Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), recoveryModulus(
          p.get<std::string>("Recovery Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), logFp(
          p.get<std::string>("logFp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), eqps(
          p.get<std::string>("Eqps Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), isoHardening(
          p.get<std::string>("IsoHardening Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    //int worksetSize = dims[0];
    std::size_t worksetSize = dims[0];

    this->addDependentField(defgrad);
    this->addDependentField(J);
    this->addDependentField(elasticModulus);
    // PoissonRatio not used in 1D stress calc
    if (numDims > 1) this->addDependentField(poissonsRatio);
    this->addDependentField(yieldStrength);
    this->addDependentField(hardeningModulus);
    this->addDependentField(recoveryModulus);

    logFpName = p.get<std::string>("logFp Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";
    isoHardeningName = p.get<std::string>("IsoHardening Name") + "_old";

    this->addEvaluatedField(stress);
    this->addEvaluatedField(logFp);
    this->addEvaluatedField(eqps);
    this->addEvaluatedField(isoHardening);

    // scratch space FCs
    //Fpinv.resize(worksetSize, numQPs, numDims, numDims);
    //FpinvT.resize(worksetSize, numQPs, numDims, numDims);
    //Cpinv.resize(worksetSize, numQPs, numDims, numDims);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void RIHMR<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(defgrad, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(elasticModulus, fm);
    if (numDims > 1) this->utils.setFieldData(poissonsRatio, fm);
    this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(yieldStrength, fm);
    this->utils.setFieldData(recoveryModulus, fm);
    this->utils.setFieldData(logFp, fm);
    this->utils.setFieldData(eqps, fm);
    this->utils.setFieldData(isoHardening, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void RIHMR<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
  {

    typedef Intrepid::FunctionSpaceTools FST;
    typedef Intrepid::RealSpaceTools<ScalarT> RST;

    ScalarT kappa, Rd;
    ScalarT mu, mubar;
    ScalarT K, Y;
    ScalarT Jm23;
    ScalarT trd3, smag;
    ScalarT Phi, p, dgam, isoH;
    ScalarT sq23 = std::sqrt(2. / 3.);

    Albany::MDArray logFpold = (*workset.stateArrayPtr)[logFpName];

    // scratch space FCs
    Intrepid::Tensor<ScalarT> be(3);
    Intrepid::Tensor<ScalarT> s(3);
    Intrepid::Tensor<ScalarT> n(3);
    Intrepid::Tensor<ScalarT> A(3);
    Intrepid::Tensor<ScalarT> expA(3);

    Intrepid::Tensor<ScalarT> Fp(3);
    Intrepid::Tensor<ScalarT> Fpold(3);
    Intrepid::Tensor<ScalarT> Fpinv(3);
    Intrepid::Tensor<ScalarT> FpinvT(3);
    Intrepid::Tensor<ScalarT> Cpinv(3);

    //Albany::MDArray Fpold = (*workset.stateArrayPtr)[fpName];
    Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];
    Albany::MDArray isoHardeningold = (*workset.stateArrayPtr)[isoHardeningName];

    // compute Cp_{n}^{-1}
    //RST::inverse(Fpinv, Fpold);
    //RST::transpose(FpinvT, Fpinv);
    //FST::tensorMultiplyDataData<ScalarT>(Cpinv, Fpinv, FpinvT);

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {

        // compute Cp_{n}^{-1}
        int cell_int = int(cell);
        int qp_int = int(qp);
        Intrepid::Tensor<ScalarT> logFp_n(logFpold(cell_int, qp_int, 0, 0),
            logFpold(cell_int, qp_int, 0, 1), logFpold(cell_int, qp_int, 0, 2),
            logFpold(cell_int, qp_int, 1, 0), logFpold(cell_int, qp_int, 1, 1),
            logFpold(cell_int, qp_int, 1, 2), logFpold(cell_int, qp_int, 2, 0),
            logFpold(cell_int, qp_int, 2, 1), logFpold(cell_int, qp_int, 2, 2));

        Fp = Intrepid::exp(logFp_n);
        Fpold = Fp;
        Fpinv = Intrepid::inverse(Fp);
        FpinvT = Intrepid::transpose(Fpinv);
        Cpinv = Intrepid::dot(Fpinv, FpinvT);

        // local parameters
        kappa = elasticModulus(cell, qp)
            / (3. * (1. - 2. * poissonsRatio(cell, qp)));
        mu = elasticModulus(cell, qp) / (2. * (1. + poissonsRatio(cell, qp)));
        K = hardeningModulus(cell, qp);
        Y = yieldStrength(cell, qp);
        Jm23 = std::pow(J(cell, qp), -2. / 3.);
        Rd = recoveryModulus(cell, qp);

        be.clear();
        // Compute Trial State
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            for (std::size_t p = 0; p < numDims; ++p)
              for (std::size_t q = 0; q < numDims; ++q)
                be(i, j) += Jm23 * defgrad(cell, qp, i, p) * Cpinv(p, q)
                    * defgrad(cell, qp, j, q);
        //be(i, j) += Jm23 * defgrad(cell, qp, i, p)
        //* Cpinv(cell, qp, p, q) * defgrad(cell, qp, j, q);

        trd3 = Intrepid::trace(be) / 3.;
        mubar = trd3 * mu;
        s = mu * (be - trd3 * Intrepid::identity<ScalarT>(3));

        isoH = isoHardeningold(cell, qp);

        // check for yielding
        smag = Intrepid::norm(s);
        Phi = smag - sq23 * (Y + isoH);

        if (Phi > 1e-11) { // plastic yielding

          // return mapping algorithm
          bool converged = false;
          int iter = 0;

          ScalarT normR0 = 0.0, conv = 0.0, normR = 0.0;
          std::vector<ScalarT> R(2);
          std::vector<ScalarT> X(2);
          std::vector<ScalarT> dRdX(4);

          dgam = 0.0;

          // initialize local unknown vector
          X[0] = dgam;
          X[1] = isoH;

          LocalNonlinearSolver<EvalT, Traits> solver;

          while (!converged) {

            compute_ResidJacobian(X, R, dRdX, isoH, smag, mubar, mu, kappa, K,
                Y, Rd);

            normR = R[0] * R[0] + R[1] * R[1];
            normR = std::sqrt(normR);

            if (iter == 0) normR0 = normR;
            if (normR0 != 0)
              conv = normR / normR0;
            else
              conv = normR0;

//              std::cout << iter << " " << normR << " " << conv << std::endl;
            if (conv < 1.e-11 || normR < 1.e-11) break;
            if (iter > 20) break;

//            TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
//                std::endl << "Error in return mapping, iter = " << iter << "\nres = " << normR << "\nrelres = " << conv << std::endl);

            solver.solve(dRdX, X, R);
            iter++;
          }

          // compute sensitivity information w.r.t system parameters, and pack back to X
          solver.computeFadInfo(dRdX, X, R);

          // update
          dgam = X[0];
          isoH = X[1];

          // plastic direction
          n = ScalarT(1. / smag) * s;

          // updated deviatoric stress
          s -= ScalarT(2. * mubar * dgam) * n;

          // update isoHardening
          isoHardening(cell, qp) = isoH;

          // update eqps
          eqps(cell, qp) = eqpsold(cell, qp) + sq23 * dgam;

          // exponential map to get Fp
          A = dgam * n;
          expA = Intrepid::exp<ScalarT>(A);

//          for (std::size_t i = 0; i < numDims; ++i) {
//            for (std::size_t j = 0; j < numDims; ++j) {
//              Fp(cell, qp, i, j) = 0.0;
//              for (std::size_t p = 0; p < numDims; ++p) {
//                Fp(cell, qp, i, j) += expA(i, p) * Fpold(cell, qp, p, j);
//              }
//            }
//          }
          for (std::size_t i = 0; i < numDims; ++i) {
            for (std::size_t j = 0; j < numDims; ++j) {
              Fp(i, j) = 0.0;
              for (std::size_t p = 0; p < numDims; ++p) {
                Fp(i, j) += expA(i, p) * Fpold(p, j);
              }
            }
          }
        } else {
          // set state variables to old values
          isoHardening(cell, qp) = isoHardeningold(cell, qp);
          eqps(cell, qp) = eqpsold(cell, qp);
          Fp = Fpold;
//          for (std::size_t i = 0; i < numDims; ++i)
//            for (std::size_t j = 0; j < numDims; ++j)
//              Fp(cell, qp, i, j) = Fpold(cell, qp, i, j);
        }

        logFp_n = Intrepid::log(Fp);
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            logFp(cell, qp, i, j) = logFp_n(i, j);

        // compute pressure
        p = 0.5 * kappa * (J(cell, qp) - 1 / (J(cell, qp)));

        // compute stress
        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = s(i, j) / J(cell, qp);
          }
          stress(cell, qp, i, i) += p;
        }

      }
    }
  }
//**********************************************************************
// all local functions
  template<typename EvalT, typename Traits>
  void RIHMR<EvalT, Traits>::compute_ResidJacobian(std::vector<ScalarT> & X,
      std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX,
      const ScalarT & isoH, const ScalarT & smag, const ScalarT & mubar,
      ScalarT & mu, ScalarT & kappa, ScalarT & K, ScalarT & Y, ScalarT & Rd)
  {
    ScalarT sq23 = std::sqrt(2. / 3.);
    std::vector<DFadType> Rfad(2);
    std::vector<DFadType> Xfad(2);
    std::vector<ScalarT> Xval(2);

    // initialize DFadType local unknown vector Xfad
    // Note that since Xfad is a temporary variable that gets changed within local iterations
    // when we initialize Xfad, we only pass in the values of X, NOT the system sensitivity information
    Xval[0] = Sacado::ScalarValue<ScalarT>::eval(X[0]);
    Xval[1] = Sacado::ScalarValue<ScalarT>::eval(X[1]);

    Xfad[0] = DFadType(2, 0, Xval[0]);
    Xfad[1] = DFadType(2, 1, Xval[1]);

    DFadType smagfad, Yfad, d_isoH;

    DFadType dgam = Xfad[0], isoHfad = Xfad[1];

    //I have to break down these equations, otherwise, there will be compile error
    //Q.Chen.
    // smagfad = smag - 2. * mubar * dgam;
    smagfad = mubar * dgam;
    smagfad = 2 * smagfad;
    smagfad = smag - smagfad;

    // Yfad = sq23 * (Y + isoHfad);
    Yfad = Y + isoHfad;
    Yfad = sq23 * Yfad;

    // d_isoH = (K - Rd * isoHfad) * sq23 * dgam;
    d_isoH = Rd * isoHfad;
    d_isoH = K - d_isoH;
    d_isoH = d_isoH * dgam;
    d_isoH = d_isoH * sq23;

    // local nonlinear sys of equations
    Rfad[0] = smagfad - Yfad; // Phi = smag - 2.* mubar * dgam - sq23 * (Y + isoHfad);
    Rfad[1] = isoHfad - isoH - d_isoH;

    // get ScalarT residual
    R[0] = Rfad[0].val();
    R[1] = Rfad[1].val();

    // get local Jacobian
    dRdX[0 + 2 * 0] = Rfad[0].dx(0);
    dRdX[0 + 2 * 1] = Rfad[0].dx(1);
    dRdX[1 + 2 * 0] = Rfad[1].dx(0);
    dRdX[1 + 2 * 1] = Rfad[1].dx(1);
  }
} // end LCM

