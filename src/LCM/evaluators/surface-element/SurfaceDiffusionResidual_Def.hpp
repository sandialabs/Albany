//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
SurfaceDiffusionResidual<EvalT, Traits>::SurfaceDiffusionResidual(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thickness(p.get<double>("thickness")),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepidBasis(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),
      scalarGrad(p.get<std::string>("Scalar Gradient Name"), dl->qp_vector),
      scalarJump(p.get<std::string>("Scalar Jump Name"), dl->qp_scalar),
      currentBasis(p.get<std::string>("Current Basis Name"), dl->qp_tensor),
      refDualBasis(
          p.get<std::string>("Reference Dual Basis Name"),
          dl->qp_tensor),
      refNormal(p.get<std::string>("Reference Normal Name"), dl->qp_vector),
      refArea(p.get<std::string>("Reference Area Name"), dl->qp_scalar),
      scalarResidual(
          p.get<std::string>("Surface Scalar Residual Name"),
          dl->node_scalar)
{
  this->addDependentField(scalarGrad);
  this->addDependentField(scalarJump);
  this->addDependentField(currentBasis);
  this->addDependentField(refDualBasis);
  this->addDependentField(refNormal);
  this->addDependentField(refArea);

  this->addEvaluatedField(scalarResidual);

  this->setName("Surface Scalar Residual" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numDims     = dims[2];

  numQPs = cubature->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims  = numDims - 1;
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceDiffusionResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(scalarGrad, fm);
  this->utils.setFieldData(scalarJump, fm);
  this->utils.setFieldData(currentBasis, fm);
  this->utils.setFieldData(refDualBasis, fm);
  this->utils.setFieldData(refNormal, fm);
  this->utils.setFieldData(refArea, fm);
  this->utils.setFieldData(scalarResidual, fm);

  // Allocate Temporary Views
  refValues =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numPlaneNodes, numQPs);
  refGrads = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", numPlaneNodes, numQPs, numPlaneDims);
  refPoints =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numPlaneDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(refValues, refPoints, Intrepid2::OPERATOR_VALUE);
  intrepidBasis->getValues(refGrads, refPoints, Intrepid2::OPERATOR_GRAD);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceDiffusionResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      scalarResidual(cell, node) = 0;
      for (int pt = 0; pt < numQPs; ++pt) {
        scalarResidual(cell, node) += refValues(node, pt) *
                                      scalarJump(cell, pt) * thickness *
                                      refArea(cell, pt);
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
