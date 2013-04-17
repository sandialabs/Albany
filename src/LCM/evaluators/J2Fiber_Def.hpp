//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <typeinfo>
namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  J2Fiber<EvalT, Traits>::J2Fiber(const Teuchos::ParameterList& p) :
          defgrad(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
          J(p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          gptLocation(p.get<std::string>("Integration Point Location Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
          elasticModulus(p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          poissonsRatio(p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          yieldStrength(p.get<std::string>("Yield Strength Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          hardeningModulus(p.get<std::string>("Hardening Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
//          satMod(p.get<std::string>("Saturation Modulus Name"),
//          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
//          satExp(p.get<std::string>("Saturation Exponent Name"),
//          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          stress(p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
          Fp(p.get<std::string>("Fp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
          eqps(p.get<std::string>("Eqps Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          energy_J2(p.get<std::string>("Energy_J2 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          energy_f1(p.get<std::string>("Energy_f1 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          energy_f2(p.get<std::string>("Energy_f2 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          damage_J2(p.get<std::string>("Damage_J2 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          damage_f1(p.get<std::string>("Damage_f1 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          damage_f2(p.get<std::string>("Damage_f2 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
          sat_mod(p.get<RealType>("Saturation Modulus Name")),
          sat_exp(p.get<RealType>("Saturation Exponent Name")),
          vol_J2(p.get<RealType>("vol_J2 Name")),
          xiinf_J2(p.get<RealType>("xiinf_J2 Name")),
          tau_J2(p.get<RealType>("tau_J2 Name")),
          k_f1(p.get<RealType>("k_f1 Name")),
          q_f1(p.get<RealType>("q_f1 Name")),
          vol_f1(p.get<RealType>("vol_f1 Name")),
          xiinf_f1(p.get<RealType>("xiinf_f1 Name")),
          tau_f1(p.get<RealType>("tau_f1 Name")),
          k_f2(p.get<RealType>("k_f2 Name")),
          q_f2(p.get<RealType>("q_f2 Name")),
          vol_f2(p.get<RealType>("vol_f2 Name")),
          xiinf_f2(p.get<RealType>("xiinf_f2 Name")),
          tau_f2(p.get<RealType>("tau_f2 Name")),
          direction_f1(p.get<Teuchos::Array<RealType> >("direction_f1 Values").toVector()),
          direction_f2(p.get<Teuchos::Array<RealType> >("direction_f2 Values").toVector()),
          ringCenter(p.get<Teuchos::Array<RealType> >("Ring Center Values").toVector()),
          isLocalCoord(p.get<bool>("isLocalCoord Name"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(defgrad);
    this->addDependentField(J);
    if (isLocalCoord) this->addDependentField(gptLocation);
    this->addDependentField(elasticModulus);
    this->addDependentField(poissonsRatio);
    this->addDependentField(yieldStrength);
    this->addDependentField(hardeningModulus);
//    this->addDependentField(satMod);
//    this->addDependentField(satExp);

    fpName = p.get<std::string>("Fp Name") + "_old";
    eqpsName = p.get<std::string>("Eqps Name") + "_old";

    energy_J2Name = p.get<std::string>("Energy_J2 Name") + "_old";
    energy_f1Name = p.get<std::string>("Energy_f1 Name") + "_old";
    energy_f2Name = p.get<std::string>("Energy_f2 Name") + "_old";

    this->addEvaluatedField(stress);
    this->addEvaluatedField(Fp);
    this->addEvaluatedField(eqps);

    this->addEvaluatedField(energy_J2);
    this->addEvaluatedField(energy_f1);
    this->addEvaluatedField(energy_f2);

    this->addEvaluatedField(damage_J2);
    this->addEvaluatedField(damage_f1);
    this->addEvaluatedField(damage_f2);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void J2Fiber<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(defgrad, fm);
    this->utils.setFieldData(J, fm);
    if(isLocalCoord) this->utils.setFieldData(gptLocation, fm);
    this->utils.setFieldData(elasticModulus, fm);
    this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(yieldStrength, fm);
//    this->utils.setFieldData(satMod, fm);
//    this->utils.setFieldData(satExp, fm);
    this->utils.setFieldData(Fp, fm);
    this->utils.setFieldData(eqps, fm);

    this->utils.setFieldData(energy_J2, fm);
    this->utils.setFieldData(energy_f1, fm);
    this->utils.setFieldData(energy_f2, fm);

    this->utils.setFieldData(damage_J2, fm);
    this->utils.setFieldData(damage_f1, fm);
    this->utils.setFieldData(damage_f2, fm);

    this->utils.setFieldData(poissonsRatio, fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void J2Fiber<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
  {
    // previous state
    Albany::MDArray Fpold = (*workset.stateArrayPtr)[fpName];
    Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];
    Albany::MDArray energy_J2old = (*workset.stateArrayPtr)[energy_J2Name];
    Albany::MDArray energy_f1old = (*workset.stateArrayPtr)[energy_f1Name];
    Albany::MDArray energy_f2old = (*workset.stateArrayPtr)[energy_f2Name];

    ScalarT kappa;
    ScalarT mu, mubar;
    ScalarT K, Y, siginf, delta;
    ScalarT Jm23;
    ScalarT trace, trd3;
    ScalarT smag, f, p, dgam;
    ScalarT sq23 = std::sqrt(2. / 3.);

    ScalarT alpha_J2, alpha_f1, alpha_f2;
    //ScalarT xi_J2, xi_f1, xi_f2;

    Intrepid::Tensor<ScalarT> F(numDims);
    Intrepid::Tensor<ScalarT> Fpn(numDims);
    Intrepid::Tensor<ScalarT> Cpinv(numDims);
    Intrepid::Tensor<ScalarT> be(numDims);
    Intrepid::Tensor<ScalarT> s(numDims);
    Intrepid::Tensor<ScalarT> N(numDims);
    Intrepid::Tensor<ScalarT> expA(numDims);
    Intrepid::Tensor<ScalarT> stress_J2(numDims);
    Intrepid::Tensor<ScalarT> C(numDims);
    Intrepid::Tensor<ScalarT> M1dyadM1(numDims);
    Intrepid::Tensor<ScalarT> M2dyadM2(numDims);
    Intrepid::Tensor<ScalarT> S0_f1(numDims);
    Intrepid::Tensor<ScalarT> S0_f2(numDims);
    Intrepid::Tensor<ScalarT> stress_f1(numDims);
    Intrepid::Tensor<ScalarT> stress_f2(numDims);

    Intrepid::Vector<ScalarT> M1(numDims);
    Intrepid::Vector<ScalarT> M2(numDims);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(numDims));

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // local parameters
        kappa = elasticModulus(cell, qp)
            / (3. * (1. - 2. * poissonsRatio(cell, qp)));
        mu = elasticModulus(cell, qp) / (2. * (1. + poissonsRatio(cell, qp)));
        K = hardeningModulus(cell, qp);
        Y = yieldStrength(cell, qp);
        siginf = sat_mod;
        delta  = sat_exp;
        Jm23   = std::pow(J(cell, qp), -2. / 3.);


        //F = Intrepid::Tensor<ScalarT>(3, &defgrad(cell, qp, 0, 0));
        F.fill(&defgrad(cell, qp, 0, 0));

        // Fill in Fpn and F Tensors with Fpold and defgrad
//        Fpn = Intrepid::Tensor<ScalarT>(Fpold(cell, qp, zero, zero),
//            Fpold(cell, qp, zero, one), Fpold(cell, qp, zero, two),
//            Fpold(cell, qp, one, zero), Fpold(cell, qp, one, one),
//            Fpold(cell, qp, one, two), Fpold(cell, qp, two, zero),
//            Fpold(cell, qp, two, one), Fpold(cell, qp, two, two));

        //Fpn.fill( &Fpold(cell,pt,std::size_t(0),std::size_t(0)) );
        for ( std::size_t i(0); i < numDims; ++i) {
          for ( std::size_t j(0); j < numDims; ++j) {
            Fpn(i,j) = static_cast<ScalarT>(Fpold(cell,qp,i,j));
          }
        }

        // compute Cpinv = Fpn^{-T} * Fpn
        Cpinv = Intrepid::transpose(inverse(Fpn)) * Fpn;

        // compute trial state
        be = Jm23 * F * Cpinv * Intrepid::transpose(F);
        trace = Intrepid::trace(be);
        trd3 = trace / numDims;
        mubar = trd3 * mu;

        // compute deviatoric stress in intermediate configuration
        s = mu * (be - trd3 * I);

        // check for yielding
        smag = Intrepid::norm(s);
        f = smag- sq23 * (Y + K * eqpsold(cell, qp)
                    + siginf * (1. - std::exp(-delta * eqpsold(cell, qp))));

        // if yield surface is violated, find plastic increment via return mapping alg
        if (f > 1.0e-11) {
          // return mapping algorithm
          bool converged = false;
          ScalarT g = f;
          ScalarT H = K * eqpsold(cell, qp)
              + siginf * (1. - std::exp(-delta * eqpsold(cell, qp)));
          ScalarT dg = (-2. * mubar) * (1. + H / (3. * mubar));
          ScalarT dH = 0.0;
          ScalarT alpha = 0.0;
          ScalarT res = 0.0;
          ScalarT relative_res = 0.0;
          int iter = 0;
          dgam = 0.0;

          while (true) {
            iter++;

            dgam  = dgam - g / dg;
            alpha = eqpsold(cell, qp) + sq23 * dgam;
            H     = K * alpha + siginf * (1. - std::exp(-delta * alpha));
            dH    = K + delta * siginf * std::exp(-delta * alpha);

            g     = smag - (2. * mubar * dgam + sq23 * (Y + H));
            dg    = -2. * mubar * (1. + dH / (3. * mubar));

            res = std::abs(g);
            relative_res = res / f;

//            std::cout << "Local iteration: " << iter << " "
//              << res << " " << relative_res << std::endl;

            if (res < 1.e-11 || relative_res < 1.0e-11)
              break;

            if(iter > 25)
              break;
//            TEUCHOS_TEST_FOR_EXCEPTION( count > 20, std::runtime_error,
//                std::endl << "Error in return mapping, iter = " << iter << "\nres = " << res << "\nrelres = " << relative_res << "\ng = " << g << "\ndg = " << dg << "\nalpha = " << alpha << std::endl);

          }

          // plastic direction (associative flow)
          if(smag != 0)
            N = (1. / smag) * s;
          else
            N.clear();

          // adjust deviatoric stress to account for plastic increment
          s = s - 2.0 * mubar * dgam * N;

          // update eqps
          eqps(cell, qp) = alpha;

          // exponential map to get Fp
          expA = Intrepid::exp(dgam * N);

          for (std::size_t i = 0; i < numDims; ++i) {
            for (std::size_t j = 0; j < numDims; ++j) {
              Fp(cell, qp, i, j) = 0.0;
              for (std::size_t k = 0; k < numDims; ++k) {
                Fp(cell, qp, i, j) += expA(i, k) * Fpn(k, j);
              }
            }
          }
        } else {
          // set state variables to old values
          eqps(cell, qp) = eqpsold(cell, qp);
          for (std::size_t i = 0; i < numDims; ++i)
            for (std::size_t j = 0; j < numDims; ++j)
              Fp(cell, qp, i, j) = Fpn(i, j);
        }

        // compute pressure
        p = 0.5 * kappa * (J(cell, qp) - 1. / (J(cell, qp)));

        // compute stress
        stress_J2 = s/J(cell,qp) + p * I;


        // compute energy for J2 stress
        energy_J2(cell, qp) = 0.5 * kappa
            * (0.5 * (J(cell, qp) * J(cell, qp) - 1.0) - std::log(J(cell, qp)))
            + 0.5 * mu * (trace - 3.0);

        // damage term in J2.
        alpha_J2 = energy_J2old(cell, qp);
        if (energy_J2(cell, qp) > alpha_J2) alpha_J2 = energy_J2(cell, qp);

        damage_J2(cell, qp) = xiinf_J2 * (1 - std::exp(-alpha_J2 / tau_J2));

        //-----------compute stress in Fibers

        // Right Cauchy-Green Tensor C = F^{T} * F
        C = Intrepid::dot(Intrepid::transpose(F), F);

        // Fiber orientation vectors

        // compute fiber orientation based on either local gauss point coordinates
        // or global direction
        if (isLocalCoord) {
          // compute fiber orientation based on local coordinates
          // special case of plane strain M1(3) = 0; M2(3) = 0;
          Intrepid::Vector<ScalarT> gpt(gptLocation(cell, qp, 0),
              gptLocation(cell, qp, 1), gptLocation(cell, qp, 2));

          Intrepid::Vector<ScalarT> OA(gpt(0) - ringCenter[0],
              gpt(1) - ringCenter[1], 0);

          M1 = OA / Intrepid::norm(OA);
          M2(0) = -M1(1);
          M2(1) = M1(0);
          M2(2) = M1(2);
        } else {

          for (std::size_t i = 0; i < numDims; ++i){
            M1(i) = direction_f1[i];
            M2(i) = direction_f2[i];
          }
           ScalarT normM1 = Intrepid::norm(M1);
           ScalarT normM2 = Intrepid::norm(M2);
           if(normM1 != 0.0)
        	   M1 = M1/Intrepid::norm(M1);

           if(normM2 != 0.0)
        	   M2 = M2/Intrepid::norm(M2);

        }

        // Anisotropic invariants I4 = M_{i} * C * M_{i}
        ScalarT I4_f1 = Intrepid::dot(M1, Intrepid::dot(C, M1));
        ScalarT I4_f2 = Intrepid::dot(M2, Intrepid::dot(C, M2));
        M1dyadM1 = Intrepid::dyad(M1, M1);
        M2dyadM2 = Intrepid::dyad(M2, M2);

        // undamaged stress (2nd PK stress)
        S0_f1 = (4.0 * k_f1 * (I4_f1 - 1.0)
            * std::exp(q_f1 * (I4_f1 - 1.0) * (I4_f1 - 1.0))) * M1dyadM1;
        S0_f2 = (4.0 * k_f2 * (I4_f2 - 1.0)
            * std::exp(q_f2 * (I4_f2 - 1.0) * (I4_f2 - 1.0))) * M2dyadM2;

        // compute energy for fibers
        energy_f1(cell, qp) = k_f1
            * (std::exp(q_f1 * (I4_f1 - 1.0) * (I4_f1 - 1.0)) - 1.0) / q_f1;
        energy_f2(cell, qp) = k_f2
            * (std::exp(q_f2 * (I4_f2 - 1.0) * (I4_f2 - 1.0)) - 1.0) / q_f2;

        // Cauchy stress
        stress_f1 = (1.0 / J(cell, qp))
            * Intrepid::dot(F, Intrepid::dot(S0_f1, Intrepid::transpose(F)));
        stress_f2 = (1.0 / J(cell, qp))
            * Intrepid::dot(F, Intrepid::dot(S0_f2, Intrepid::transpose(F)));

        // maximum thermodynamic forces
        alpha_f1 = energy_f1old(cell, qp);
        alpha_f2 = energy_f2old(cell, qp);

        if (energy_f1(cell, qp) > alpha_f1) alpha_f1 = energy_f1(cell, qp);

        if (energy_f2(cell, qp) > alpha_f2) alpha_f2 = energy_f2(cell, qp);

        // damage term in fibers
        damage_f1(cell, qp) = xiinf_f1 * (1.0 - std::exp(-alpha_f1 / tau_f1));
        damage_f2(cell, qp) = xiinf_f2 * (1.0 - std::exp(-alpha_f2 / tau_f2));

        // total Cauchy stress (J2, Fibers)
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            stress(cell, qp, i, j) =
                vol_J2 * (1.0 - damage_J2(cell, qp)) * stress_J2(i,j)
                + vol_f1 * (1.0 - damage_f1(cell, qp)) * stress_f1(i, j)
                + vol_f2 * (1.0 - damage_f2(cell, qp)) * stress_f2(i, j);
      } // end of loop over qp
    } // end of loop over cell
  }
} // end LCM

