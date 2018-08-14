//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_MaterialDatabase.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
DefGrad<EvalT, Traits>::DefGrad(const Teuchos::ParameterList& p)
    : GradU(
          p.get<std::string>("Gradient QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      weights(
          p.get<std::string>("Weights Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      defgrad(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      weightedAverage(false),
      alpha(0.05)
{
  if (p.isType<bool>("Weighted Volume Average J"))
    weightedAverage = p.get<bool>("Weighted Volume Average J");
  if (p.isType<RealType>("Average J Stabilization Parameter"))
    alpha = p.get<RealType>("Average J Stabilization Parameter");

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");

  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  worksetSize = dims[0];
  numQPs      = dims[1];
  numDims     = dims[2];

  this->addDependentField(GradU);
  this->addDependentField(weights);

  this->addEvaluatedField(defgrad);
  this->addEvaluatedField(J);

  this->setName("DefGrad" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
DefGrad<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights, fm);
  this->utils.setFieldData(defgrad, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(GradU, fm);
}
//**********************************************************************
template <typename EvalT, typename Traits>
void
DefGrad<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // bool print = false;
  // if (typeid(ScalarT) == typeid(RealType)) print = true;

  // Compute DefGrad tensor from displacement gradient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int i = 0; i < numDims; ++i) {
        for (int j = 0; j < numDims; ++j) {
          defgrad(cell, qp, i, j) = GradU(cell, qp, i, j);
        }
        defgrad(cell, qp, i, i) += 1.0;
      }
    }
  }
  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable
  // values. Leaving this out leads to inversion of 0 tensors.
  for (int cell = workset.numCells; cell < worksetSize; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int i = 0; i < numDims; ++i) { defgrad(cell, qp, i, i) = 1.0; }
    }
  }

  Intrepid2::RealSpaceTools<PHX::Device>::det(J.get_view(), defgrad.get_view());

  if (weightedAverage) {
    ScalarT Jbar, wJbar, vol;
    for (int cell = 0; cell < workset.numCells; ++cell) {
      Jbar = 0.0;
      vol  = 0.0;
      for (int qp = 0; qp < numQPs; ++qp) {
        Jbar += weights(cell, qp) * J(cell, qp);
        vol += weights(cell, qp);
      }
      Jbar /= vol;

      for (int qp = 0; qp < numQPs; ++qp) {
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            wJbar = std::exp(
                (1 - alpha) * std::log(Jbar) + alpha * std::log(J(cell, qp)));
            defgrad(cell, qp, i, j) *= std::cbrt(wJbar / J(cell, qp));
          }
        }
        J(cell, qp) = wJbar;
      }
    }
  }
}

//**********************************************************************
}  // namespace LCM
