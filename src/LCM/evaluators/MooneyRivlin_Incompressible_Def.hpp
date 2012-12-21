//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Tensor.h"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  MooneyRivlin_Incompressible<EvalT, Traits>::MooneyRivlin_Incompressible(
      const Teuchos::ParameterList& p) :
      F(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), c1(
          p.get<double>("c1 Name")), c2(p.get<double>("c2 Name")), mult(
          p.get<double>("mult Name"))
  {
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl = p.get<
        Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(F);
    this->addDependentField(J);

    this->addEvaluatedField(stress);

    // scratch space FCs
    FT.resize(worksetSize, numQPs, numDims, numDims);

    this->setName(
        "Incompressible MooneyRivlin Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlin_Incompressible<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(F, fm);
    this->utils.setFieldData(J, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlin_Incompressible<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    cout.precision(15);
    LCM::Tensor<ScalarT> S(3);
    LCM::Tensor<ScalarT> C_qp(3);
    LCM::Tensor<ScalarT> F_qp(3);
    LCM::Tensor<ScalarT> Cbar(3);
    LCM::Tensor<ScalarT> Svol(3);
    LCM::Tensor<ScalarT> Siso(3);
    LCM::Tensor<ScalarT> Sbar(3);
    LCM::Tensor4<ScalarT> PP(3);
    LCM::Tensor<ScalarT> Id = LCM::identity<ScalarT>(3);

    ScalarT Jm23;
    ScalarT mu = 2.0 * (c1 + c2);
    // Assume that kappa (bulk modulus) = scalar multiplier (mult) * mu (shear modulus)
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

        // eq 6.84 Holzapfel
        PP = LCM::identity_1<ScalarT>(3)
            - (1.0 / 3.0) * LCM::tensor(LCM::inverse(C_qp), C_qp);

        ScalarT pressure = kappa * (J(cell, qp) - 1);

        Jm23 = std::pow(J(cell, qp), -2.0 / 3.0);

        Cbar = Jm23 * C_qp;

        Svol = pressure * J(cell, qp) * LCM::inverse(C_qp);

        // table 6.2 Holzapfel
        ScalarT gamma_bar1 = 2.0 * (c1 + c2 * LCM::I1(Cbar));
        ScalarT gamma_bar2 = -2.0 * c2;

        Sbar = gamma_bar1 * Id + gamma_bar2 * Cbar;
        Siso = Jm23 * LCM::dotdot(PP, Sbar);

        S = Svol + Siso; // decomposition of stress tensor per Holzapfel

        // Convert to Cauchy stress
        S = (1. / J(cell, qp)) * F_qp * S * LCM::transpose(F_qp);

        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = S(i, j);
          }
        }
      }
    }
  }

//**********************************************************************
}

