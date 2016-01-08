//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#include <Intrepid2_MiniTensor.h>
#include <Sacado_MathFunctions.hpp>

#include <typeinfo>

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  ThermoMechanicalStress<EvalT, Traits>::ThermoMechanicalStress(
      const Teuchos::ParameterList& p) :
      F_array(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")), J_array(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), shearModulus(
          p.get<std::string>("Shear Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), bulkModulus(
          p.get<std::string>("Bulk Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), temperature(
          p.get<std::string>("Temperature Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), yieldStrength(
          p.get<std::string>("Yield Strength Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), hardeningModulus(
          p.get<std::string>("Hardening Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), satMod(
          p.get<std::string>("Saturation Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), satExp(
          p.get<std::string>("Saturation Exponent Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), deltaTime(
          p.get<std::string>("Delta Time Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")), Fp(
          p.get<std::string>("Fp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")), eqps(
          p.get<std::string>("eqps Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), mechSource(
          p.get<std::string>("Mechanical Source Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")), thermalExpansionCoeff(
          p.get<RealType>("Thermal Expansion Coefficient")), refTemperature(
          p.get<RealType>("Reference Temperature"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];

    this->addDependentField(F_array);
    this->addDependentField(J_array);
    this->addDependentField(shearModulus);
    this->addDependentField(bulkModulus);
    this->addDependentField(yieldStrength);
    this->addDependentField(hardeningModulus);
    this->addDependentField(satMod);
    this->addDependentField(satExp);
    this->addDependentField(temperature);
    this->addDependentField(deltaTime);

    fpName = p.get<std::string>("Fp Name") + "_old";
    eqpsName = p.get<std::string>("eqps Name") + "_old";
    this->addEvaluatedField(stress);
    this->addEvaluatedField(Fp);
    this->addEvaluatedField(eqps);
    this->addEvaluatedField(mechSource);

    this->setName("ThermoMechanical Stress" + PHX::typeAsString<EvalT>());

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void ThermoMechanicalStress<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(Fp, fm);
    this->utils.setFieldData(eqps, fm);
    this->utils.setFieldData(F_array, fm);
    this->utils.setFieldData(J_array, fm);
    this->utils.setFieldData(shearModulus, fm);
    this->utils.setFieldData(bulkModulus, fm);
    this->utils.setFieldData(temperature, fm);
    this->utils.setFieldData(hardeningModulus, fm);
    this->utils.setFieldData(satMod, fm);
    this->utils.setFieldData(satExp, fm);
    this->utils.setFieldData(yieldStrength, fm);
    this->utils.setFieldData(mechSource, fm);
    this->utils.setFieldData(deltaTime, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void ThermoMechanicalStress<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    bool print = false;
    //if (typeid(ScalarT) == typeid(RealType)) print = true;

    if (print) std::cout << " *** ThermoMechanicalStress *** " << std::endl;

    // declare some ScalarT's to be used later
    ScalarT J, Jm23, K, H, Y, siginf, delta;
    ScalarT f, dgam;
    ScalarT deltaTemp;
    ScalarT mu, mubar;
    ScalarT smag;
    ScalarT pressure;
    ScalarT sq23 = std::sqrt(2. / 3.);

    // local Tensors
    Intrepid2::Tensor<ScalarT> F(3), Fpold(3), Fpinv(3), Cpinv(3);
    Intrepid2::Tensor<ScalarT> be(3), s(3), N(3), A(3), expA(3);

    // grab the time step
    ScalarT dt = deltaTime(0);

    // get old state variables
    Albany::MDArray Fpold_array = (*workset.stateArrayPtr)[fpName];
    Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // Fill in tensors from MDArray data
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            Fpold(i, j) = Fpold_array(cell, qp, i, j);
            F(i, j) = F_array(cell, qp, i, j);
          }
        }

        // local qp values (for readibility)
        J = J_array(cell, qp);
        Jm23 = std::pow(J, -2. / 3.);
        mu = shearModulus(cell, qp);
        K = bulkModulus(cell, qp);
        H = hardeningModulus(cell, qp);
        Y = yieldStrength(cell, qp);
        siginf = satMod(cell, qp);
        delta = satExp(cell, qp);
        deltaTemp = temperature(cell, qp) - refTemperature;

        // initialize plastic work
        mechSource(cell, qp) = 0.0;

        // compute the pressure
        pressure = 0.5 * K
            * ((J - 1 / J)
                - 3 * thermalExpansionCoeff * deltaTemp * (1 + 1 / (J * J)));

        // compute trial intermediate configuration
        Fpinv = Intrepid2::inverse(Fpold);
        Cpinv = Fpinv * Intrepid2::transpose(Fpinv);
        be = F * Cpinv * Intrepid2::transpose(F);

        // compute the trial deviatoric stress
        mubar = ScalarT(Intrepid2::trace(be) / 3.) * mu;
        s = mu * Intrepid2::dev(be);

        smag = Intrepid2::norm(s);
        f = smag
            - sq23
                * (Y + H * eqpsold(cell, qp)
                    + siginf * (1. - std::exp(-delta * eqpsold(cell, qp))));

        dgam = 0.0;

        if (f > 1E-6) {
          // return mapping algorithm
          bool converged = false;
          ScalarT g = f;
          ScalarT G = H * eqpsold(cell, qp)
              + siginf * (1. - std::exp(-delta * eqpsold(cell, qp)));
          ScalarT dg = (-2. * mubar) * (1. + H / (3. * mubar));
          ScalarT dG = 0.0;
          ;
          ScalarT alpha = 0.0;
          ScalarT res = 0.0;
          int count = 0;

          while (!converged && count < 50) {
            count++;

            dgam -= g / dg;

            alpha = eqpsold(cell, qp) + sq23 * dgam;

            G = H * alpha + siginf * (1. - std::exp(-delta * alpha));
            ;
            dG = H + delta * siginf * std::exp(-delta * alpha);
            ;

            g = smag - (2. * mubar * dgam + sq23 * (Y + G));
            dg = -2. * mubar * (1. + dG / (3. * mubar));

            res = std::abs(g);
            if (res < 1.e-6 || res / f < 1.e-6) converged = true;

            TEUCHOS_TEST_FOR_EXCEPTION( count > 50, std::runtime_error,
                std::endl << "Error in return mapping, count = " << count << "\nres = " << res << "\nrelres = " << res/f << "\ng = " << g << "\ndg = " << dg << "\nalpha = " << alpha << std::endl);

          }

          // plastic direction
          N = ScalarT(1 / smag) * s;

          // updated deviatoric stress
          s -= ScalarT(2. * mubar * dgam) * N;

          // update eqps
          eqps(cell, qp) = alpha;

          // exponential map to get Fp
          A = dgam * N;
         expA = Intrepid2::exp<ScalarT>(A);

          // set plastic work
          if (dt > 0.0) mechSource(cell, qp) = sq23 * dgam / dt
              * (Y + G + temperature(cell, qp) * 1.0);

          for (int i = 0; i < numDims; ++i) {
            for (int j = 0; j < numDims; ++j) {
              Fp(cell, qp, i, j) = 0.0;
              for (int p = 0; p < numDims; ++p) {
                Fp(cell, qp, i, j) += expA(i, p) * Fpold(p, j);
              }
            }
          }
        } else {
          // set state variables to old values
          //dp(cell, qp) = 0.0;
          eqps(cell, qp) = eqpsold(cell, qp);
          for (int i = 0; i < numDims; ++i)
            for (int j = 0; j < numDims; ++j)
              Fp(cell, qp, i, j) = Fpold_array(cell, qp, i, j);
        }

        // compute stress
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = s(i, j) / J;
          }
          stress(cell, qp, i, i) += pressure;
        }

        // update be
      be = ScalarT(1 / mu) * s +
            ScalarT(Intrepid2::trace(be) / 3) * Intrepid2::eye<ScalarT>(3);

        if (print) {
          std::cout << "    sig : ";
          for (unsigned int i(0); i < numDims; ++i)
            for (unsigned int j(0); j < numDims; ++j)
              std::cout << stress(cell, qp, i, j) << " ";
          std::cout << std::endl;

          std::cout << "    s   : ";
          for (unsigned int i(0); i < numDims; ++i)
            for (unsigned int j(0); j < numDims; ++j)
              std::cout << s(i, j) << " ";
          std::cout << std::endl;

          std::cout << "    work: " << mechSource(cell, qp) << std::endl;
          std::cout << "    dgam: " << dgam << std::endl;
          std::cout << "    smag: " << smag << std::endl;
          std::cout << "    n(s): " << Intrepid2::norm(s) << std::endl;
          std::cout << "    temp: " << temperature(cell, qp) << std::endl;
          std::cout << "    Dtem: " << deltaTemp << std::endl;
          std::cout << "       Y: " << yieldStrength(cell, qp) << std::endl;
          std::cout << "       H: " << hardeningModulus(cell, qp) << std::endl;
          std::cout << "       S: " << satMod(cell, qp) << std::endl;
        }
      }
    }
  }
//**********************************************************************

}
