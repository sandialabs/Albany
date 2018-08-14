//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
PeridigmPartialStressBase<EvalT, Traits>::PeridigmPartialStressBase(
    const Teuchos::ParameterList& p)
    : J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      defgrad(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout"))
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Input
  this->addDependentField(J);
  this->addDependentField(defgrad);

  // Output
  this->addEvaluatedField(stress);

  this->setName("PeridigmPartialStress" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
PeridigmPartialStressBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(defgrad, fm);
  this->utils.setFieldData(stress, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
PeridigmPartialStressBase<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  bool albanyIsCreatingMassMatrix = true;
  if (workset.m_coeff != 0.0) { albanyIsCreatingMassMatrix = false; }
  if (workset.j_coeff != 0.0) { albanyIsCreatingMassMatrix = false; }
  if (workset.n_coeff != -1.0) { albanyIsCreatingMassMatrix = false; }
  if (!albanyIsCreatingMassMatrix) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        "PeridigmPartialStressBase::evaluateFields not implemented for this "
        "template type.",
        Teuchos::Exceptions::InvalidParameter,
        "Need specialization.");
  }
}

//**********************************************************************
template <typename Traits>
void
PeridigmPartialStress<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  std::string      blockName       = workset.EBName;
  int              worksetIndex    = static_cast<int>(workset.wsIndex);
  PeridigmManager& peridigmManager = *PeridigmManager::self();

  // Container for the partial stress values at each quadrature point in an
  // element
  std::vector<std::vector<RealType>> partialStressValues;
  partialStressValues.resize(this->numQPs);
  for (int i = 0; i < this->numQPs; ++i) partialStressValues[i].resize(9);

  RealType detJ, defGradTranspose[3][3], piolaStress[3][3], cauchyStress[3][3];

  for (int cell = 0; cell < workset.numCells; ++cell) {
    peridigmManager.getPartialStress(
        blockName, worksetIndex, cell, partialStressValues);
    for (int qp = 0; qp < this->numQPs; ++qp) {
      this->stress(cell, qp, 0, 0) = partialStressValues[qp][0];
      this->stress(cell, qp, 1, 1) = partialStressValues[qp][4];
      this->stress(cell, qp, 2, 2) = partialStressValues[qp][8];
      this->stress(cell, qp, 0, 1) = partialStressValues[qp][1];
      this->stress(cell, qp, 1, 2) = partialStressValues[qp][5];
      this->stress(cell, qp, 2, 0) = partialStressValues[qp][6];
      this->stress(cell, qp, 1, 0) = partialStressValues[qp][3];
      this->stress(cell, qp, 2, 1) = partialStressValues[qp][7];
      this->stress(cell, qp, 0, 2) = partialStressValues[qp][2];

      //       piolaStress[0][0] = partialStressValues[qp][0];
      //       piolaStress[0][1] = partialStressValues[qp][1];
      //       piolaStress[0][2] = partialStressValues[qp][2];
      //       piolaStress[1][0] = partialStressValues[qp][3];
      //       piolaStress[1][1] = partialStressValues[qp][4];
      //       piolaStress[1][2] = partialStressValues[qp][5];
      //       piolaStress[2][0] = partialStressValues[qp][6];
      //       piolaStress[2][1] = partialStressValues[qp][7];
      //       piolaStress[2][2] = partialStressValues[qp][8];

      //       detJ = this->J(cell,qp);

      //       defGradTranspose[0][0] = this->defgrad(cell,qp,0,0);
      //       defGradTranspose[1][0] = this->defgrad(cell,qp,0,1);
      //       defGradTranspose[2][0] = this->defgrad(cell,qp,0,2);
      //       defGradTranspose[0][1] = this->defgrad(cell,qp,1,0);
      //       defGradTranspose[1][1] = this->defgrad(cell,qp,1,1);
      //       defGradTranspose[2][1] = this->defgrad(cell,qp,1,2);
      //       defGradTranspose[0][2] = this->defgrad(cell,qp,2,0);
      //       defGradTranspose[1][2] = this->defgrad(cell,qp,2,1);
      //       defGradTranspose[2][2] = this->defgrad(cell,qp,2,2);

      //       cauchyStress[0][0] = (1.0/detJ) *
      //       (piolaStress[0][0]*defGradTranspose[0][0] +
      //       piolaStress[0][1]*defGradTranspose[1][0] +
      //       piolaStress[0][2]*defGradTranspose[2][0]); cauchyStress[0][1] =
      //       (1.0/detJ) * (piolaStress[0][0]*defGradTranspose[0][1] +
      //       piolaStress[0][1]*defGradTranspose[1][1] +
      //       piolaStress[0][2]*defGradTranspose[2][1]); cauchyStress[0][2] =
      //       (1.0/detJ) * (piolaStress[0][0]*defGradTranspose[0][2] +
      //       piolaStress[0][1]*defGradTranspose[1][2] +
      //       piolaStress[0][2]*defGradTranspose[2][2]); cauchyStress[1][0] =
      //       (1.0/detJ) * (piolaStress[1][0]*defGradTranspose[0][0] +
      //       piolaStress[1][1]*defGradTranspose[1][0] +
      //       piolaStress[1][2]*defGradTranspose[2][0]); cauchyStress[1][1] =
      //       (1.0/detJ) * (piolaStress[1][0]*defGradTranspose[0][1] +
      //       piolaStress[1][1]*defGradTranspose[1][1] +
      //       piolaStress[1][2]*defGradTranspose[2][1]); cauchyStress[1][2] =
      //       (1.0/detJ) * (piolaStress[1][0]*defGradTranspose[0][2] +
      //       piolaStress[1][1]*defGradTranspose[1][2] +
      //       piolaStress[1][2]*defGradTranspose[2][2]); cauchyStress[2][0] =
      //       (1.0/detJ) * (piolaStress[2][0]*defGradTranspose[0][0] +
      //       piolaStress[2][1]*defGradTranspose[1][0] +
      //       piolaStress[2][2]*defGradTranspose[2][0]); cauchyStress[2][1] =
      //       (1.0/detJ) * (piolaStress[2][0]*defGradTranspose[0][1] +
      //       piolaStress[2][1]*defGradTranspose[1][1] +
      //       piolaStress[2][2]*defGradTranspose[2][1]); cauchyStress[2][2] =
      //       (1.0/detJ) * (piolaStress[2][0]*defGradTranspose[0][2] +
      //       piolaStress[2][1]*defGradTranspose[1][2] +
      //       piolaStress[2][2]*defGradTranspose[2][2]);

      //       this->stress(cell,qp,0,0) = cauchyStress[0][0];
      //       this->stress(cell,qp,1,1) = cauchyStress[1][1];
      //       this->stress(cell,qp,2,2) = cauchyStress[2][2];
      //       this->stress(cell,qp,0,1) = cauchyStress[0][1];
      //       this->stress(cell,qp,1,2) = cauchyStress[1][2];
      //       this->stress(cell,qp,2,0) = cauchyStress[2][0];
      //       this->stress(cell,qp,1,0) = cauchyStress[1][0];
      //       this->stress(cell,qp,2,1) = cauchyStress[2][1];
      //       this->stress(cell,qp,0,2) = cauchyStress[0][2];
    }
  }

  //   for(int cell=0; cell < workset.numCells; ++cell){
  //     peridigmManager.getPartialStress(blockName, worksetIndex, cell,
  //     partialStressValues); for (int qp=0; qp < this->numQPs; ++qp) {
  //       this->stress(cell,qp,0,0) = 0.0;
  //       this->stress(cell,qp,1,1) = 0.0;
  //       this->stress(cell,qp,2,2) = 0.0;
  //       this->stress(cell,qp,0,1) = 0.0;
  //       this->stress(cell,qp,1,2) = 0.0;
  //       this->stress(cell,qp,2,0) = 0.0;
  //       this->stress(cell,qp,1,0) = 0.0;
  //       this->stress(cell,qp,2,1) = 0.0;
  //       this->stress(cell,qp,0,2) = 0.0;
  //     }
  //   }
}

//**********************************************************************
}  // namespace LCM
