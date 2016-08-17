//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "LocalNonlinearSolver.hpp"

#include <typeinfo>

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  J2Damage<EvalT, Traits>::J2Damage(const Teuchos::ParameterList& p) :
      defgrad(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), bulkModulus(
          p.get<std::string>("Bulk Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), shearModulus(
          p.get<std::string>("Shear Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), yieldStrength(
          p.get<std::string>("Yield Strength Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), hardeningModulus(
          p.get<std::string>("Hardening Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), satMod(
          p.get<std::string>("Saturation Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), satExp(
          p.get<std::string>("Saturation Exponent Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), damage(
          p.get<std::string>("Damage Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")), dp(
          p.get<std::string>("DP Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), seff(
          p.get<std::string>("Effective Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), energy(
          p.get<std::string>("Energy Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), Fp(
          p.get<std::string>("Fp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")), eqps(
          p.get<std::string>("Eqps Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(defgrad);
    this->addDependentField(J);
    this->addDependentField(bulkModulus);
    this->addDependentField(shearModulus);
    this->addDependentField(yieldStrength);
    this->addDependentField(hardeningModulus);
    this->addDependentField(satMod);
    this->addDependentField(satExp);
    this->addDependentField(damage);

    fpName = p.get<std::string>("Fp Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";
    this->addEvaluatedField(stress);
    this->addEvaluatedField(dp);
    this->addEvaluatedField(seff);
    this->addEvaluatedField(energy);
    this->addEvaluatedField(Fp);
    this->addEvaluatedField(eqps);

    this->setName("Stress" + PHX::typeAsString<EvalT>());

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void J2Damage<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(dp, fm);
    this->utils.setFieldData(seff, fm);
    this->utils.setFieldData(energy, fm);
    this->utils.setFieldData(defgrad, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(bulkModulus, fm);
    this->utils.setFieldData(shearModulus, fm);
    this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(yieldStrength, fm);
    this->utils.setFieldData(satMod, fm);
    this->utils.setFieldData(satExp, fm);
    this->utils.setFieldData(damage, fm);
    this->utils.setFieldData(Fp, fm);
    this->utils.setFieldData(eqps, fm);

    // scratch space FCs
    Fpinv = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    FpinvT = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    Cpinv = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void J2Damage<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    LocalNonlinearSolver<EvalT, Traits> solver;

    bool print = false;
    //if (typeid(ScalarT) == typeid(RealType)) print = true;

    typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
    typedef Intrepid2::RealSpaceTools<PHX::Device> RST;

    ScalarT kappa, H, H2, phi, phi_old;
    ScalarT mu, mubar;
    ScalarT K, Y, siginf, delta;
    ScalarT Jm23;
    ScalarT trd3;
    ScalarT smag, f, p, dgam;
    ScalarT sq23 = std::sqrt(2. / 3.);

    // scratch space FCs
    Intrepid2::Tensor<ScalarT> be(3);
    Intrepid2::Tensor<ScalarT> s(3);
    Intrepid2::Tensor<ScalarT> N(3);
    Intrepid2::Tensor<ScalarT> A(3);
    Intrepid2::Tensor<ScalarT> expA(3);

    Albany::MDArray Fpold_shards = (*workset.stateArrayPtr)[fpName];
    Albany::MDArray eqpsold_shards = (*workset.stateArrayPtr)[eqpsName];
    Albany::MDArray phi_old_FC_shards = (*workset.stateArrayPtr)["Damage_old"];

    // Copy Shards MDArray into Kokkos DynRankView containers
    Kokkos::DynRankView<RealType, PHX::Device> Fpold("STA",workset.numCells, numQPs, numDims, numDims);
    Kokkos::DynRankView<RealType, PHX::Device> eqpsold("STA",workset.numCells, numQPs);
    Kokkos::DynRankView<RealType, PHX::Device> phi_old_FC("STA",workset.numCells, numQPs);

    for (int cell=0; cell < workset.numCells; ++cell)
      for (int qp=0; qp < numQPs; ++qp)
        for (int i=0; i < numDims; ++i)
          for (int j=0; j < numDims; ++j)
            Fpold(cell,qp,i,j)=Fpold_shards(cell,qp,i,j);

    for (int cell=0; cell < workset.numCells; ++cell)
      for (int qp=0; qp < numQPs; ++qp)
        eqpsold(cell,qp) = eqpsold_shards(cell,qp);

    for (int cell=0; cell < workset.numCells; ++cell)
      for (int qp=0; qp < numQPs; ++qp)
        phi_old_FC(cell,qp) = phi_old_FC_shards(cell,qp);

    // compute Cp_{n}^{-1}
//std::cout << "AGS: commenting out necessary line to get code to compile. states need to be Kokkos, or copied to Kokkos" << std::endl;
    RST::inverse(Fpinv, Fpold);
    RST::transpose(FpinvT, Fpinv);
    FST::tensorMultiplyDataData(Cpinv, Fpinv, FpinvT);

    // std::cout << "F:\n";
    // for (int cell=0; cell < workset.numCells; ++cell)
    // {
    //   for (int qp=0; qp < numQPs; ++qp)
    //   {
    //     for (int i=0; i < numDims; ++i)
    //    for (int j=0; j < numDims; ++j)
    //      std::cout << Sacado::ScalarValue<ScalarT>::eval(defgrad(cell,qp,i,j)) << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // std::cout << "Fpold:\n";
    // for (int cell=0; cell < workset.numCells; ++cell)
    // {
    //   for (int qp=0; qp < numQPs; ++qp)
    //   {
    //     for (int i=0; i < numDims; ++i)
    //    for (int j=0; j < numDims; ++j)
    //      std::cout << Sacado::ScalarValue<ScalarT>::eval(Fpold(cell,qp,i,j)) << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // std::cout << "Cpinv:\n";
    // for (int cell=0; cell < workset.numCells; ++cell)
    // {
    //   for (int qp=0; qp < numQPs; ++qp)
    //   {
    //     for (int i=0; i < numDims; ++i)
    //    for (int j=0; j < numDims; ++j)
    //      std::cout << Sacado::ScalarValue<ScalarT>::eval(Cpinv(cell,qp,i,j)) << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl;

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // local parameters
        kappa = bulkModulus(cell, qp);
        mu = shearModulus(cell, qp);
        K = hardeningModulus(cell, qp);
        Y = yieldStrength(cell, qp);
        siginf = satMod(cell, qp);
        delta = satExp(cell, qp);
        Jm23 = std::pow(J(cell, qp), -2. / 3.);
        phi_old = phi_old_FC(cell, qp);
        //phi    = std::max( phi_old, damage(cell,qp) );
        phi = damage(cell, qp);
        H = (1.0 - phi);
        H2 = H * H;

        be.clear();
        // Compute Trial State
        for (int i = 0; i < numDims; ++i)
          for (int j = 0; j < numDims; ++j)
            for (int p = 0; p < numDims; ++p)
              for (int q = 0; q < numDims; ++q)
                be(i, j) += Jm23 * defgrad(cell, qp, i, p)
                    * Cpinv(cell, qp, p, q) * defgrad(cell, qp, j, q);

        // std::cout << "be: \n" << be;

        trd3 = Intrepid2::trace(be) / 3.;
        mubar = trd3 * mu;
        s = mu * (be - trd3 * Intrepid2::eye<ScalarT>(3));

        // std::cout << "s: \n" << s;

        // check for yielding
        smag = norm(s);
        //f = smag - exph * sq23 * ( Y + K * eqpsold(cell,qp) + siginf * ( 1. - std::exp( -delta * eqpsold(cell,qp) ) ) );
        f = smag
            - sq23
                * (Y + K * eqpsold(cell, qp)
                    + siginf * (1. - std::exp(-delta * eqpsold(cell, qp))));

        // std::cout << "smag : " << Sacado::ScalarValue<ScalarT>::eval(smag) << std::endl;
        // std::cout << "eqpsold: " << Sacado::ScalarValue<ScalarT>::eval(eqpsold(cell,qp)) << std::endl;
        // std::cout << "K      : " << Sacado::ScalarValue<ScalarT>::eval(K) << std::endl;
        // std::cout << "Y      : " << Sacado::ScalarValue<ScalarT>::eval(Y) << std::endl;
        // std::cout << "f      : " << Sacado::ScalarValue<ScalarT>::eval(f) << std::endl;

        if (f > 1E-8) {
          // return mapping algorithm
          bool converged = false;
          ScalarT g = f;
          ScalarT H = K * eqpsold(cell, qp)
              + siginf * (1. - std::exp(-delta * eqpsold(cell, qp)));
          ScalarT dg = (-2. * mubar) * (1. + H / (3. * mubar));
          ScalarT dH = 0.0;
          ;
          ScalarT alpha = 0.0;
          ScalarT res = 0.0;
          int count = 0;
          dgam = 0.0;

          while (!converged) {
            count++;

            //dgam = ( f / ( 2. * mubar) ) / ( 1. + K / ( 3. * mubar ) );
            dgam -= g / dg;

            alpha = eqpsold(cell, qp) + sq23 * dgam;

            //H = K * alpha + exph * siginf * ( 1. - std::exp( -delta * alpha ) );
            //dH = K + exph * delta * siginf * std::exp( -delta * alpha );
            H = K * alpha + siginf * (1. - std::exp(-delta * alpha));
            dH = K + delta * siginf * std::exp(-delta * alpha);

            g = smag - (2. * mubar * dgam + sq23 * (Y + H));
            dg = -2. * mubar * (1. + dH / (3. * mubar));

            res = std::abs(g);
            if (res < 1.e-8 || res / f < 1.e-8) converged = true;

            TEUCHOS_TEST_FOR_EXCEPTION( count > 50, std::runtime_error,
                std::endl << "Error in return mapping, count = " << count << "\nres = " << res << "\nrelres = " << res/f << "\ng = " << g << "\ndg = " << dg << "\nalpha = " << alpha << std::endl);

          }

          // set dp for this QP
          dp(cell, qp) = dgam;

          // plastic direction
          N = ScalarT(1. / smag) * s;

          // updated deviatoric stress
          s -= ScalarT(2. * mubar * dgam) * N;

          // update eqps
          eqps(cell, qp) = alpha;

          // exponential map to get Fp
          A = dgam * N;
          expA = Intrepid2::exp<ScalarT>(A);

          // std::cout << "expA: \n";
          // for (int i=0; i < numDims; ++i)
          //   for (int j=0; j < numDims; ++j)
          //     std::cout << Sacado::ScalarValue<ScalarT>::eval(expA(i,j)) << " ";
          // std::cout << std::endl;

          for (int i = 0; i < numDims; ++i) {
            for (int j = 0; j < numDims; ++j) {
              Fp(cell, qp, i, j) = 0.0;
              for (int p = 0; p < numDims; ++p) {
                Fp(cell, qp, i, j) += expA(i, p) * Fpold(cell, qp, p, j);
              }
            }
          }
        } else {
          // set state variables to old values
          dp(cell, qp) = 0.0;
          eqps(cell, qp) = eqpsold(cell, qp);
          for (int i = 0; i < numDims; ++i)
            for (int j = 0; j < numDims; ++j)
              Fp(cell, qp, i, j) = Fpold(cell, qp, i, j);
        }

        // compute pressure
        p = 0.5 * kappa * (J(cell, qp) - 1 / (J(cell, qp)));

        // compute stress
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = s(i, j) / J(cell, qp);
            //stress(cell,qp,i,j) = s(i,j) / Je;
          }
          stress(cell, qp, i, i) += p;
        }

        // scale stress by damage
        for (int i = 0; i < numDims; ++i)
          for (int j = 0; j < numDims; ++j)
            stress(cell, qp, i, j) *= H2;

        // update be
        be = ScalarT(1 / mu) * s + trd3 * Intrepid2::eye<ScalarT>(3);

        // compute energy
        energy(cell, qp) = 0.5 * kappa
            * (0.5 * (J(cell, qp) * J(cell, qp) - 1.0) - std::log(J(cell, qp)))
            + 0.5 * mu * (Intrepid2::trace(be) - 3.0);

        // compute seff for damage coupling
        seff(cell, qp) = Intrepid2::norm(ScalarT(1.0 / J(cell, qp)) * s);

        if (print) {
          std::cout << "********" << std::endl;
          std::cout << "damage : " << damage(cell, qp) << std::endl;
          std::cout << "phi    : " << phi << std::endl;
          std::cout << "H2     : " << H2 << std::endl;
          std::cout << "stress : ";
          for (int i = 0; i < numDims; ++i)
            for (int j = 0; j < numDims; ++j)
              std::cout << stress(cell, qp, i, j) << " ";
          std::cout << std::endl;
          std::cout << "energy : " << energy(cell, qp) << std::endl;
          std::cout << "dp     : " << dp(cell, qp) << std::endl;
        }
      }
    }
  }
//**********************************************************************
}// end LCM

