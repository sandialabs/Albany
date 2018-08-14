//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
LatticeDefGrad<EvalT, Traits>::LatticeDefGrad(const Teuchos::ParameterList& p)
    : weights(
          p.get<std::string>("Weights Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      defgrad(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      JH(p.get<std::string>("DetDefGradH Name"),
         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      CtotalRef(
          p.get<std::string>("Stress Free Total Concentration Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      Ctotal(
          p.get<std::string>("Total Concentration Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      VH(p.get<std::string>("Partial Molar Volume Name"),
         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      VM(p.get<std::string>("Molar Volume Name"),
         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      latticeDefGrad(
          p.get<std::string>("Lattice Deformation Gradient Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      weightedAverage(false),
      alpha(0.05)
{
  if (p.isType<std::string>("Weighted Volume Average J Name"))
    weightedAverage = p.get<bool>("Weighted Volume Average J");
  if (p.isType<double>("Average J Stabilization Parameter Name"))
    alpha = p.get<double>("Average J Stabilization Parameter");

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");

  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  worksetSize = dims[0];
  numQPs      = dims[1];
  numDims     = dims[2];

  this->addDependentField(weights);
  this->addDependentField(CtotalRef);
  this->addDependentField(Ctotal);
  this->addDependentField(VH);
  this->addDependentField(VM);
  this->addDependentField(defgrad);
  this->addDependentField(J);

  this->addEvaluatedField(latticeDefGrad);
  this->addEvaluatedField(JH);

  this->setName("Lattice Deformation Gradient" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
LatticeDefGrad<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights, fm);
  this->utils.setFieldData(defgrad, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(JH, fm);
  this->utils.setFieldData(CtotalRef, fm);
  this->utils.setFieldData(Ctotal, fm);
  this->utils.setFieldData(VH, fm);
  this->utils.setFieldData(VM, fm);
  this->utils.setFieldData(latticeDefGrad, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
LatticeDefGrad<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // Compute LatticeDefGrad tensor from displacement gradient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int i = 0; i < numDims; ++i) {
        for (int j = 0; j < numDims; ++j) {
          latticeDefGrad(cell, qp, i, j) = defgrad(cell, qp, i, j);
        }
      }
      JH(cell, qp) = J(cell, qp);
    }
  }
  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable
  // values. Leaving this out leads to inversion of 0 tensors.
  for (int cell = workset.numCells; cell < worksetSize; ++cell)
    for (int qp = 0; qp < numQPs; ++qp)
      for (int i = 0; i < numDims; ++i) latticeDefGrad(cell, qp, i, i) = 1.0;

  if (weightedAverage) {
    ScalarT Jbar, wJbar, vol;
    for (int cell = 0; cell < workset.numCells; ++cell) {
      Jbar = 0.0;
      vol  = 0.0;
      for (int qp = 0; qp < numQPs; ++qp) {
        Jbar +=
            weights(cell, qp) *
            std::log(
                1 + VH(cell, qp) * (Ctotal(cell, qp) - CtotalRef(cell, qp)));
        vol += weights(cell, qp);
      }
      Jbar /= vol;
      // Jbar = std::exp(Jbar);
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            wJbar = std::exp(
                (1 - alpha) * Jbar +
                alpha * std::log(
                            1 + VH(cell, qp) *
                                    (Ctotal(cell, qp) - CtotalRef(cell, qp))));
            latticeDefGrad(cell, qp, i, j) *= std::pow(wJbar, -1. / 3.);
          }
        }
        JH(cell, qp) *= wJbar;
      }
    }
  } else {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        JH(cell, qp) *=
            (1 + VH(cell, qp) * (Ctotal(cell, qp) - CtotalRef(cell, qp)));
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
            latticeDefGrad(cell, qp, i, j) *= std::pow(JH(cell, qp), -1. / 3.);
          }
        }
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
