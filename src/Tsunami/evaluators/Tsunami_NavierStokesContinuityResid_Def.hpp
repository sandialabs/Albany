//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace Tsunami {


//**********************************************************************
template<typename EvalT, typename Traits>
NavierStokesContinuityResid<EvalT, Traits>::
NavierStokesContinuityResid(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF                (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  VGrad              (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_tensor),
  CResidual          (p.get<std::string> ("Residual Name"), dl->node_scalar),
  havePSPG           (p.get<bool>("Have PSPG"))
{
  this->addDependentField(wBF);
  this->addDependentField(VGrad);
  if (havePSPG) {
    wGradBF = decltype(wGradBF)(
      p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector);
    TauPSPG = decltype(TauPSPG)(
      p.get<std::string>("Tau Name"), dl->qp_scalar);
    Rm = decltype(Rm)(
      p.get<std::string>("Rm Name"), dl->qp_vector);
    densityQP = decltype(densityQP)(
      p.get<std::string> ("Fluid Density QP Name"), dl->qp_scalar);
    this->addDependentField(wGradBF);
    this->addDependentField(TauPSPG);
    this->addDependentField(Rm);
    this->addDependentField(densityQP);
  }

  this->addEvaluatedField(CResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numCells = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("NavierStokesContinuityResid"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NavierStokesContinuityResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(VGrad,fm);
  if (havePSPG) {
    this->utils.setFieldData(wGradBF,fm);
    this->utils.setFieldData(TauPSPG,fm);
    this->utils.setFieldData(Rm,fm);
    this->utils.setFieldData(densityQP,fm);
  }

  this->utils.setFieldData(CResidual,fm);

  // Allocate workspace
  divergence = Kokkos::createDynRankView(VGrad.get_view(), "XXX", numCells, numQPs);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void NavierStokesContinuityResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      divergence(cell,qp) = 0.0;
      for (std::size_t i=0; i < numDims; ++i) {
        divergence(cell,qp) += VGrad(cell,qp,i,i);
      }
    }
  }
  FST::integrate(CResidual.get_view(), divergence, wBF.get_view(),
                          false); // "false" overwrites

  if (havePSPG) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t j=0; j < numDims; ++j) {
            CResidual(cell,node) += TauPSPG(cell,qp)/densityQP(cell,qp)*Rm(cell,qp,j)*wGradBF(cell,node,qp,j);
          }
        }
      }
    }
  }

}

//**********************************************************************
}

