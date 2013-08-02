//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  MooneyRivlin<EvalT, Traits>::MooneyRivlin(const Teuchos::ParameterList& p) :
      F(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), c1(
          p.get<double>("c1 Name")), c2(p.get<double>("c2 Name")), c(
          p.get<double>("c Name"))
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

    this->setName("MooneyRivlin Stress" + PHX::TypeString<EvalT>::value);

  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlin<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(F, fm);
    this->utils.setFieldData(J, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void MooneyRivlin<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    std::cout.precision(15);
    Intrepid::Tensor<ScalarT> S(3);
    Intrepid::Tensor<ScalarT> C_qp(3);
    Intrepid::Tensor<ScalarT> F_qp(3);

    ScalarT d = 2.0 * (c1 + 2 * c2);

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

        S = 2.0 * (c1 + c2 * Intrepid::I1(C_qp)) * Intrepid::identity<ScalarT>(3)
            - 2.0 * c2 * C_qp
            + (2.0 * c * J(cell, qp) * (J(cell, qp) - 1) - d)
                * Intrepid::inverse(C_qp);

        // Convert to Cauchy stress
        S = (1. / J(cell, qp)) * F_qp * S * Intrepid::transpose(F_qp);

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

