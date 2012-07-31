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
#include "LocalNonlinearSolver.h"

#include <typeinfo>

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  RIHMR<EvalT, Traits>::RIHMR(const Teuchos::ParameterList& p) :
      defgrad(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      elasticModulus(p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      poissonsRatio(p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      yieldStrength(p.get<std::string>("Yield Strength Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      hardeningModulus(p.get<std::string>("Hardening Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      recoveryModulus(p.get<std::string>("Recovery Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      stress(p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
      Fp(p.get<std::string>("Fp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
      eqps(p.get<std::string>("Eqps Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
      ess(p.get<std::string>("Ess Name"),
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

    fpName = p.get<std::string>("Fp Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";
    essName = p.get<std::string>("Ess Name") + "_old";

    this->addEvaluatedField(stress);
    this->addEvaluatedField(Fp);
    this->addEvaluatedField(eqps);
    this->addEvaluatedField(ess);

    // scratch space FCs
    Fpinv.resize(worksetSize, numQPs, numDims, numDims);
    FpinvT.resize(worksetSize, numQPs, numDims, numDims);
    Cpinv.resize(worksetSize, numQPs, numDims, numDims);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void RIHMR<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(defgrad, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(elasticModulus, fm);
    if (numDims > 1) this->utils.setFieldData(poissonsRatio, fm);
    this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(yieldStrength, fm);
    this->utils.setFieldData(recoveryModulus, fm);
    this->utils.setFieldData(Fp, fm);
    this->utils.setFieldData(eqps, fm);
    this->utils.setFieldData(ess, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void RIHMR<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {

    typedef Intrepid::FunctionSpaceTools FST;
    typedef Intrepid::RealSpaceTools<ScalarT> RST;

    ScalarT kappa, Rd;
    ScalarT mu, mubar;
    ScalarT K, Y;
    ScalarT Jm23;
    ScalarT trd3, smag;
    ScalarT Phi, p, dgam, es;
    ScalarT sq23 = std::sqrt(2. / 3.);

    Albany::MDArray Fpold = (*workset.stateArrayPtr)[fpName];
    Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];
    Albany::MDArray essold = (*workset.stateArrayPtr)[essName];

    // compute Cp_{n}^{-1}
    RST::inverse(Fpinv, Fpold);
    RST::transpose(FpinvT, Fpinv);
    FST::tensorMultiplyDataData<ScalarT>(Cpinv, Fpinv, FpinvT);

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // local parameters
        kappa = elasticModulus(cell, qp)
              / (3. * (1. - 2. * poissonsRatio(cell, qp)));
        mu = elasticModulus(cell, qp) / (2. * (1. + poissonsRatio(cell, qp)));
        K = hardeningModulus(cell, qp);
        Y = yieldStrength(cell, qp);
        Jm23 = std::pow(J(cell, qp), -2. / 3.);
        Rd = recoveryModulus(cell,qp);

        be.clear();
        // Compute Trial State
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            for (std::size_t p = 0; p < numDims; ++p)
              for (std::size_t q = 0; q < numDims; ++q)
                be(i, j) += Jm23 * defgrad(cell, qp, i, p)
                    * Cpinv(cell, qp, p, q) * defgrad(cell, qp, j, q);

        trd3 = trace(be) / 3.;
        mubar = trd3 * mu;
        s = mu * (be - trd3 * LCM::identity<ScalarT, 3>());

        es = essold(cell,qp);

        // check for yielding
        smag = LCM::norm(s);
        Phi = smag - sq23 * (Y + 2. * mu * es);

//        std::cout << "Rd      : " << Sacado::ScalarValue<ScalarT>::eval(Rd) << std::endl;

        if (Phi > 1e-11) { // plastic yielding

          // return mapping algorithm
          bool converged = false;
          int iter = 0;

          ScalarT normR0=0.0, conv = 0.0, normR = 0.0;
          std::vector<ScalarT> R(2);
          std::vector<ScalarT> X(2);
          std::vector<ScalarT>dRdX(4);

          dgam = 0.0;

          // initialize local unkown vector
          X[0]  = dgam; X[1] = es;

          LocalNonlinearSolver<EvalT, Traits> solver;

          while (!converged) {

        	  compute_ResidJacobian(X, R, dRdX, es, smag, mubar, mu, kappa, K, Y, Rd);

        	  normR = R[0]*R[0] + R[1]* R[1];
        	  normR = std::sqrt(normR);

        	  if(iter == 0) normR0 = normR;
        	  if(normR0 != 0)
        		  conv = normR / normR0;
        	  else
        		  conv = normR0;

//              std::cout << iter << " " << normR << " " << conv << std::endl;
              if (conv < 1.e-11 || normR < 1.e-11) break;
              if(iter > 20) break;

//            TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
//                std::endl << "Error in return mapping, iter = " << iter << "\nres = " << normR << "\nrelres = " << conv << std::endl);

        	  solver.solve(dRdX, X, R);
        	  iter++;
          }

          // compute sensitivity information w.r.t system parameters, and pack back to X
          solver.computeFadInfo(dRdX, X, R);

          // update
          dgam = X[0];
          es = X[1];

          // plastic direction
          n = ScalarT(1. / smag) * s;

          // updated deviatoric stress
          s -= ScalarT(2. * mubar * dgam) * n;

          // update ess
          ess(cell, qp) = es;

          // update eqps
          eqps(cell, qp) = eqpsold(cell,qp) + sq23 * dgam;

          // exponential map to get Fp
          A = dgam * n;
          expA = LCM::exp<ScalarT>(A);

          for (std::size_t i = 0; i < numDims; ++i) {
            for (std::size_t j = 0; j < numDims; ++j) {
              Fp(cell, qp, i, j) = 0.0;
              for (std::size_t p = 0; p < numDims; ++p) {
                Fp(cell, qp, i, j) += expA(i, p) * Fpold(cell, qp, p, j);
              }
            }
          }
        } else {
          // set state variables to old values
          ess(cell, qp) = essold(cell,qp);
          eqps(cell, qp) = eqpsold(cell, qp);
          for (std::size_t i = 0; i < numDims; ++i)
            for (std::size_t j = 0; j < numDims; ++j)
              Fp(cell, qp, i, j) = Fpold(cell, qp, i, j);
        }

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
      std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX, const ScalarT & es,
      const ScalarT & smag, const ScalarT & mubar, ScalarT & mu, ScalarT & kappa, ScalarT & K, ScalarT & Y, ScalarT & Rd)
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

    DFadType smagfad, Yfad, des;

    DFadType dgam = Xfad[0], esfad = Xfad[1];

    //I have to break down these equations, otherwise, there will be compile error
    //Q.Chen.
    // smagfad = smag - 2. * mubar * dgam;
    smagfad = mubar * dgam;
    smagfad = 2 * smagfad;
    smagfad = smag - smagfad;

	// Yfad = sq23 * (Y + 2. * mu *esfad);
    Yfad = mu * esfad;
    Yfad = 2. * Yfad;
    Yfad = Y + Yfad;
    Yfad = sq23 * Yfad;

    // des = (K - Rd * esfad) * sq23 * dgam;
    des = Rd * esfad;
    des = K - des;
    des = des * dgam;
    des = des * sq23;

    // local nonlinear sys of equations
    Rfad[0] = smagfad - Yfad; // Phi = smag - 2.* mubar * dgam - sq23 * (Y + 2. * mu * esfad);
    Rfad[1] = esfad - es - des;

    // get ScalarT residual
    R[0] = Rfad[0].val();
    R[1] = Rfad[1].val();

    // get local Jacobian
    dRdX[0 + 2 * 0] = Rfad[0].dx(0);
    dRdX[0 + 2 * 1] = Rfad[0].dx(1);
    dRdX[1 + 2 * 0] = Rfad[1].dx(0);
    dRdX[1 + 2 * 1] = Rfad[1].dx(1);
  }
}// end LCM

