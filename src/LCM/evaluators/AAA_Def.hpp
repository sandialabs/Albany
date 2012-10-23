//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "LCM/utils/Tensor.h"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  AAA<EvalT, Traits>::AAA(const Teuchos::ParameterList& p) :
      F(p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), J(
          p.get<std::string>("DetDefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), alpha(
          p.get<double>("alpha Name")), beta(p.get<double>("beta Name")), mult(
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

    this->setName("Incompressible AAA Stress" + PHX::TypeString<EvalT>::value);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void AAA<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(F, fm);
    this->utils.setFieldData(J, fm);
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
  void AAA<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
  {
    cout.precision(15);
    LCM::Tensor<ScalarT> S(3);
    LCM::Tensor<ScalarT> B_qp(3);
    LCM::Tensor<ScalarT> Id = LCM::identity<ScalarT>(3);

    ScalarT Jm23;
    //per Rajagopal and Tao, Journal of Elasticity 28(2) (1992), 165-184
    ScalarT mu = 2.0 * (alpha);
    // Assume that kappa (bulk modulus) = scalar multiplier (mult) * mu (shear modulus)
    ScalarT kappa = mult * mu;

    Intrepid::FieldContainer<ScalarT> B(worksetSize, numQPs, numDims, numDims);
    Intrepid::RealSpaceTools<ScalarT>::transpose(FT, F);
    // Left Cauchy-Green deformation tensor
    Intrepid::FunctionSpaceTools::tensorMultiplyDataData<ScalarT>(B, F, FT,
        'N');

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        B_qp.clear();
        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            B_qp(i, j) = B(cell, qp, i, j);
          }
        }

        ScalarT pressure = kappa * (J(cell, qp) - 1);

        // Cauchy stress
        S = -pressure * Id
            + 2.0 * (alpha + 2.0 * beta * (LCM::I1(B_qp) - 3.0)) * B_qp;

        for (std::size_t i = 0; i < numDims; ++i) {
          for (std::size_t j = 0; j < numDims; ++j) {
            stress(cell, qp, i, j) = S(i, j);
          }
        }
      }
    }
  }
} //namespace LCM
