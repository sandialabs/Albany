//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  AnisotropicHyperelasticDamage<EvalT, Traits>::
  AnisotropicHyperelasticDamage(const Teuchos::ParameterList& p,
                                const Teuchos::RCP<Albany::Layouts>& dl) :
    defgrad(p.get<std::string>("DefGrad Name"),dl->qp_tensor),
    J(p.get<std::string>("DetDefGrad Name"),dl->qp_scalar),
    elasticModulus(p.get<std::string>("Elastic Modulus Name"),dl->qp_scalar),
    poissonsRatio(p.get<std::string>("Poissons Ratio Name"),dl->qp_scalar),
    coordVec(p.get<std::string>("QP Coordinate Vector Name"),dl->qp_vector),
    stress(p.get<std::string>("Stress Name"),dl->qp_tensor),
    energyM(p.get<std::string>("EnergyM Name"),dl->qp_scalar),
    energyF1(p.get<std::string>("EnergyF1 Name"),dl->qp_scalar),
    energyF2(p.get<std::string>("EnergyF2 Name"),dl->qp_scalar),
    damageM(p.get<std::string>("DamageM Name"),dl->qp_scalar),
    damageF1(p.get<std::string>("DamageF1 Name"),dl->qp_scalar),
    damageF2(p.get<std::string>("DamageF2 Name"),dl->qp_scalar)
  {
    // Pull out numQPs and numDims from a Layout
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(defgrad);
    this->addDependentField(J);
    this->addDependentField(elasticModulus);
    this->addDependentField(poissonsRatio);
    this->addDependentField(coordVec);

    energyMName = p.get<std::string>("EnergyM Name") + "_old";
    energyF1Name = p.get<std::string>("EnergyF1 Name") + "_old";
    energyF2Name = p.get<std::string>("EnergyF2 Name") + "_old";

    this->addEvaluatedField(stress);
    this->addEvaluatedField(energyM);
    this->addEvaluatedField(energyF1);
    this->addEvaluatedField(energyF2);
    this->addEvaluatedField(damageM);
    this->addEvaluatedField(damageF1);
    this->addEvaluatedField(damageF2);

    this->setName("Stress" + PHX::TypeString<EvalT>::value);

    // get parameter list of material constants
    pList = p.get<Teuchos::ParameterList*>("Parameter List");

    // set material parameters
    kF1 = pList->get<RealType>("Fiber 1 k");
    qF1 = pList->get<RealType>("Fiber 1 q");
    volFracF1    = pList->get<RealType>("Fiber 1 volume fraction");
    damageMaxF1  = pList->get<RealType>("Fiber 1 maximum damage");
    saturationF1 = pList->get<RealType>("Fiber 1 damage saturation"); 

    kF2 = pList->get<RealType>("Fiber 2 k");
    qF2 = pList->get<RealType>("Fiber 2 q");
    volFracF2    = pList->get<RealType>("Fiber 2 volume fraction");
    damageMaxF2  = pList->get<RealType>("Fiber 2 maximum damage");
    saturationF2 = pList->get<RealType>("Fiber 2 damage saturation"); 

    volFracM    = pList->get<RealType>("Matrix volume fraction");
    damageMaxM  = pList->get<RealType>("Matrix maximum damage");
    saturationM = pList->get<RealType>("Matrix damage saturation"); 
    
    // check for volume fraction sanity
    std::string msg="In Anisotropic Hyperelastic damage -- "
      "Volume Fraction of Matrix and Fibers not equal to one";
    TEUCHOS_TEST_FOR_EXCEPTION(!(volFracM+volFracF1+volFracF2==1.0),
                               Teuchos::Exceptions::InvalidParameter,
                               msg);

    directionF1 = 
      pList->get<Teuchos::Array<RealType> >("Fiber 1 Orientation Vector").toVector();
    directionF2 = 
      pList->get<Teuchos::Array<RealType> >("Fiber 2 Orientation Vector").toVector();
    isLocalCoord = pList->get<bool>("Use Local Coordinate System",false);
    if (isLocalCoord)
      ringCenter = 
        pList->get<Teuchos::Array<RealType> >("Ring Center Vector").toVector();
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void AnisotropicHyperelasticDamage<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d, 
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(defgrad, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(coordVec, fm);
    this->utils.setFieldData(elasticModulus, fm);
    this->utils.setFieldData(poissonsRatio, fm);

    this->utils.setFieldData(energyM, fm);
    this->utils.setFieldData(energyF1, fm);
    this->utils.setFieldData(energyF2, fm);

    this->utils.setFieldData(damageM, fm);
    this->utils.setFieldData(damageF1, fm);
    this->utils.setFieldData(damageF2, fm);

    this->utils.setFieldData(stress, fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void AnisotropicHyperelasticDamage<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
  {

    //std::cout << "In AnisotropicHyperelasticDamage evaluate Fields" << std::endl;

    ScalarT kappa,mu,Jm53,Jm23,p;
    ScalarT alphaF1, alphaF2, alphaM;
    //ScalarT xi_M, xi_F1, xi_F2;

    // Define some tensors for use
    LCM::Tensor<ScalarT> I(LCM::eye<ScalarT>(numDims));
    LCM::Tensor<ScalarT> F(numDims), s(numDims), b(numDims), C(numDims);
    LCM::Tensor<ScalarT> sigmaM(numDims), sigmaF1(numDims), sigmaF2(numDims);

    // previous state
    Albany::MDArray energyMold = (*workset.stateArrayPtr)[energyMName];
    Albany::MDArray energyF1old = (*workset.stateArrayPtr)[energyF1Name];
    Albany::MDArray energyF2old = (*workset.stateArrayPtr)[energyF2Name];

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // local parameters
        kappa = elasticModulus(cell, qp)
          / (3. * (1. - 2. * poissonsRatio(cell, qp)));
        mu = elasticModulus(cell, qp) / (2. * (1. + poissonsRatio(cell, qp)));
        Jm53 = std::pow(J(cell, qp), -5. / 3.);
        Jm23 = std::pow(J(cell, qp), -2. / 3.);
        F = LCM::Tensor<ScalarT>(3, &defgrad(cell, qp, 0, 0));
        
        // compute deviatoric stress
        b = F*LCM::transpose(F);
        s = mu * Jm53 * LCM::dev(b);
        // compute pressure
        p = 0.5 * kappa * (J(cell, qp) - 1. / (J(cell, qp)));

        sigmaM = s + p*I;

        // compute energy for M
        energyM(cell, qp) = 0.5 * kappa
          * (0.5 * (J(cell, qp) * J(cell, qp) - 1.0) - std::log(J(cell, qp)))
          + 0.5 * mu * ( Jm23*LCM::trace(b) - 3.0);

        // damage term in M.
        alphaM = energyMold(cell, qp);
        if (energyM(cell, qp) > alphaM) alphaM = energyM(cell, qp);

        damageM(cell, qp) = damageMaxM * (1 - std::exp(-alphaM / saturationM));

        //-----------compute stress in Fibers

        // Right Cauchy-Green Tensor C = F^{T} * F
        C = LCM::dot(LCM::transpose(F), F);

        // Fiber orientation vectors
        LCM::Vector<ScalarT> M1(0.0, 0.0, 0.0);
        LCM::Vector<ScalarT> M2(0.0, 0.0, 0.0);

        // compute fiber orientation based on either local gauss point coordinates
        // or global direction
        if (isLocalCoord) {
          // compute fiber orientation based on local coordinates
          // special case of plane strain M1(3) = 0; M2(3) = 0;
          LCM::Vector<ScalarT> gpt(coordVec(cell, qp, 0),
                                   coordVec(cell, qp, 1), 
                                   coordVec(cell, qp, 2));

          LCM::Vector<ScalarT> OA(gpt(0) - ringCenter[0],
                                  gpt(1) - ringCenter[1], 0);

          M1 = OA / norm(OA);
          M2(0) = -M1(1);
          M2(1) = M1(0);
          M2(2) = M1(2);
        } else {
          M1(0) = directionF1[0];
          M1(1) = directionF1[1];
          M1(2) = directionF1[2];
          M2(0) = directionF2[0];
          M2(1) = directionF2[1];
          M2(2) = directionF2[2];
        }

        // Anisotropic invariants I4 = M_{i} * C * M_{i}
        ScalarT I4F1 = LCM::dot(M1, LCM::dot(C, M1));
        ScalarT I4F2 = LCM::dot(M2, LCM::dot(C, M2));
        LCM::Tensor<ScalarT> M1dyadM1 = dyad(M1, M1);
        LCM::Tensor<ScalarT> M2dyadM2 = dyad(M2, M2);

        // undamaged stress (2nd PK stress)
        LCM::Tensor<ScalarT> S0F1(3), S0F2(3);
        S0F1 = (4.0 * kF1 * (I4F1 - 1.0)
                * std::exp(qF1 * (I4F1 - 1) * (I4F1 - 1))) * M1dyadM1;
        S0F2 = (4.0 * kF2 * (I4F2 - 1.0)
                * std::exp(qF2 * (I4F2 - 1) * (I4F2 - 1))) * M2dyadM2;

        // compute energy for fibers
        energyF1(cell, qp) = kF1
          * (std::exp(qF1 * (I4F1 - 1) * (I4F1 - 1)) - 1) / qF1;
        energyF2(cell, qp) = kF2
          * (std::exp(qF2 * (I4F2 - 1) * (I4F2 - 1)) - 1) / qF2;

        // Fiber Cauchy stress
        sigmaF1 = (1.0 / J(cell, qp))
          * LCM::dot(F, LCM::dot(S0F1, LCM::transpose(F)));
        sigmaF2 = (1.0 / J(cell, qp))
          * LCM::dot(F, LCM::dot(S0F2, LCM::transpose(F)));

        // maximum thermodynamic forces
        alphaF1 = energyF1old(cell, qp);
        alphaF2 = energyF2old(cell, qp);

        if (energyF1(cell, qp) > alphaF1) alphaF1 = energyF1(cell, qp);

        if (energyF2(cell, qp) > alphaF2) alphaF2 = energyF2(cell, qp);

        // damage term in fibers
        damageF1(cell, qp) = damageMaxF1 * (1 - std::exp(-alphaF1 / saturationF1));
        damageF2(cell, qp) = damageMaxF2 * (1 - std::exp(-alphaF2 / saturationF2));

        // total Cauchy stress (M, Fibers)
        for (std::size_t i = 0; i < numDims; ++i)
          for (std::size_t j = 0; j < numDims; ++j)
            stress(cell, qp, i, j) = volFracM * (1 - damageM(cell, qp)) * sigmaM(i, j)
              + volFracF1 * (1 - damageF1(cell, qp)) * sigmaF1(i, j)
              + volFracF2 * (1 - damageF2(cell, qp)) * sigmaF2(i, j);

      } // end of loop over qp
    } // end of loop over cell
  }
} // end LCM

