//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM
{

//**********************************************************************
  template<typename EvalT, typename Traits>
  MooneyRivlin_Incompressible_Damage<EvalT, Traits>::MooneyRivlin_Incompressible_Damage(
      const Teuchos::ParameterList& p) :
      F(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), alpha(
          p.get<std::string>("alpha Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), c1(
          p.get<double>("c1 Name")), c2(p.get<double>("c2 Name")), mult(
          p.get<double>("mult Name")), zeta_inf(p.get<double>("zeta_inf Name")), iota(
          p.get<double>("iota Name"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    alphaName = p.get<std::string>("alpha Name") + "_old";

    this->addDependentField(F);
    this->addDependentField(J);

    this->addEvaluatedField(stress);
    this->addEvaluatedField(alpha);

    // scratch space FCs
    FT.resize(worksetSize, numQPs, numDims, numDims);

    this->setName(
        "Incompressible MooneyRivlin Stress" + PHX::TypeString<EvalT>::value);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlin_Incompressible_Damage<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(F, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(alpha, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlin_Incompressible_Damage<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {

    Albany::MDArray alphaold = (*workset.stateArrayPtr)[alphaName];

    cout.precision(15);
    Intrepid::Tensor<ScalarT> S(3);
    Intrepid::Tensor<ScalarT> C_qp(3);
    Intrepid::Tensor<ScalarT> F_qp(3);
    Intrepid::Tensor<ScalarT> Cbar(3);
    Intrepid::Tensor<ScalarT> Cinv(3);
    Intrepid::Tensor<ScalarT> Svol(3);
    Intrepid::Tensor<ScalarT> Siso(3);
    Intrepid::Tensor<ScalarT> Sbar(3);
    Intrepid::Tensor4<ScalarT> PP(3);
    Intrepid::Tensor4<ScalarT> PPbar(3);
    Intrepid::Tensor4<ScalarT> SS = (1.0 / 2.0)
        * (Intrepid::identity_1<ScalarT>(3) + Intrepid::identity_2<ScalarT>(3));
    Intrepid::Tensor4<ScalarT> CCvol(3);
    Intrepid::Tensor4<ScalarT> CCbar(3);
    Intrepid::Tensor4<ScalarT> CCiso(3);
    Intrepid::Tensor4<ScalarT> CC(3); // full elasticity tensor
    Intrepid::Tensor<ScalarT> Id = Intrepid::identity<ScalarT>(3);

    ScalarT Jm23;
    ScalarT mu = 2.0 * (c1 + c2);
    // Assume that kappa (bulk modulus) =
    // scalar multiplier (mult) * mu (shear modulus)
    ScalarT kappa = mult * mu;

    Intrepid::FieldContainer<ScalarT> C(worksetSize, numQPs, numDims, numDims);
    Intrepid::RealSpaceTools<ScalarT>::transpose(FT, F);
    Intrepid::FunctionSpaceTools::tensorMultiplyDataData<ScalarT>(C, FT, F,
        'N');

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        C_qp.clear();
        F_qp.clear();
        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            C_qp(i, j) = C(cell, qp, i, j);
            F_qp(i, j) = F(cell, qp, i, j);
          }
        }

        // Per Holzapfel, a scalar damage model is added to the
        // strain energy function to model isotropic damage
        // Compute the strain energy at the current step
        ScalarT Psi_0 = c1 * (Intrepid::I1(C_qp) - 3.0)
            + c2 * (Intrepid::I2(C_qp) - 3.0);

        // as the max function is not defined for this variable type
        ScalarT alphaold_comp = alphaold(cell, qp);

        alpha(cell, qp) = std::max(alphaold_comp, Psi_0);
        ScalarT zeta = zeta_inf * (1.0 - std::exp(-(alpha(cell, qp) / iota)));

        Cinv = Intrepid::inverse(C_qp);

        // eq 6.84 Holzapfel
        PP = Intrepid::identity_1<ScalarT>(3)
            - (1.0 / 3.0) * Intrepid::tensor(Cinv, C_qp);

        ScalarT pressure = kappa * (J(cell, qp) - 1);

        Jm23 = std::pow(J(cell, qp), -2.0 / 3.0);

        Cbar = Jm23 * C_qp;

        Svol = pressure * J(cell, qp) * Cinv;

        // table 6.2 Holzapfel
        ScalarT gamma_bar1 = 2.0 * (c1 + c2 * Intrepid::I1(Cbar));
        ScalarT gamma_bar2 = -2.0 * c2;

        Sbar = gamma_bar1 * Id + gamma_bar2 * Cbar;
        // damage only affects the isochoric stress
        Siso = (1.0 - zeta) * Jm23 * Intrepid::dotdot(PP, Sbar);

        S = Svol + Siso; // decomposition of stress tensor per Holzapfel

        // Convert to Cauchy stress
        S = (1. / J(cell, qp)) * F_qp * S * Intrepid::transpose(F_qp);

        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = S(i, j);
          }
        }

        // Compute the elasticity tensor
        ScalarT pressure_bar = pressure + J(cell, qp) * kappa;
        CCvol = J(cell, qp) * pressure_bar * Intrepid::tensor(Cinv, Cinv)
            - 2.0 * J(cell, qp) * pressure * Intrepid::odot(Cinv, Cinv);

        PPbar = Intrepid::odot(Cinv, Cinv)
            - (1.0 / 3.0) * Intrepid::tensor(Cinv, Cinv);

        CCbar = 16.0 * c2 * std::pow(J(cell, qp), -4. / 3.)
            * (Intrepid::tensor(Id, Id) - SS);

        Intrepid::Tensor4<ScalarT> PPCCbar = Intrepid::dotdot(PP, CCbar);
        CCiso = Intrepid::dotdot(PPCCbar, Intrepid::transpose(PP))
            + (2. / 3.) * std::pow(J(cell, qp), -4. / 3.)
                * Intrepid::dotdot(Sbar, C_qp) * PPbar
            - (2. / 3.)
                * (Intrepid::tensor(Cinv, Siso) + Intrepid::tensor(Siso, Cinv));

        // As in the stress, the damage only affects the
        // isochoric portion of the elasticity tensor
        CC = CCvol + (1.0 - zeta) * CCiso;

      }
    }
  }
} //namespace LCM
