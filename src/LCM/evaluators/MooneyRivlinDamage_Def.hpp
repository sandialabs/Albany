//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "VectorTensorBase.h"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  MooneyRivlinDamage<EvalT, Traits>::MooneyRivlinDamage(
      const Teuchos::ParameterList& p) :
      F(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), alpha(
          p.get<std::string>("alpha Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), c1(
          p.get<double>("c1 Name")), c2(p.get<double>("c2 Name")), c(
          p.get<double>("c Name")), zeta_inf(p.get<double>("zeta_inf Name")), iota(
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

    this->setName("MooneyRivlinDamage Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlinDamage<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(F, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(alpha, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlinDamage<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    cout.precision(15);
    LCM::Tensor<ScalarT> S(3);
    LCM::Tensor<ScalarT> C_qp(3);
    LCM::Tensor<ScalarT> F_qp(3);

    ScalarT d = 2.0 * (c1 + 2.0 * c2);

    Albany::MDArray alphaold = (*workset.stateArrayPtr)[alphaName];

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

        // Per Holzapfel, a scalar damage model is added to the strain energy function to model isotropic damage
        // Compute the strain energy at the current step
        ScalarT Psi_0 = c * std::pow((J(cell, qp) - 1), 2.0)
            - d * std::log(J(cell, qp)) + c1 * (LCM::I1(C_qp) - 3.0)
            + c2 * (LCM::I2(C_qp) - 3.0);

        ScalarT alphaold_comp = alphaold(cell, qp); // as the max function is not defined for this variable type

        ScalarT zeta = zeta_inf * (1.0 - std::exp(-(alpha(cell, qp) / iota)));

        S = 2.0 * (c1 + c2 * LCM::I1(C_qp)) * LCM::identity<ScalarT>(3)
            - 2.0 * c2 * C_qp
            + (2 * c * J(cell, qp) * (J(cell, qp) - 1.0) - d)
                * LCM::inverse(C_qp);
        S = (1.0 - zeta) * S; // Damage the material

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

