//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
SurfaceScalarGradient<EvalT, Traits>::SurfaceScalarGradient(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thickness(p.get<double>("thickness")),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepidBasis(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),
      refDualBasis(
          p.get<std::string>("Reference Dual Basis Name"),
          dl->qp_tensor),
      refNormal(p.get<std::string>("Reference Normal Name"), dl->qp_vector),
      jump(p.get<std::string>("Scalar Jump Name"), dl->qp_scalar),
      nodalScalar(p.get<std::string>("Nodal Scalar Name"), dl->node_scalar),
      scalarGrad(
          p.get<std::string>("Surface Scalar Gradient Name"),
          dl->qp_vector)
{
  this->addDependentField(refDualBasis);
  this->addDependentField(refNormal);
  this->addDependentField(jump);
  this->addDependentField(nodalScalar);

  this->addEvaluatedField(scalarGrad);

  this->setName("Surface Scalar Gradient" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numDims     = dims[2];

  numQPs = cubature->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims  = numDims - 1;

#ifdef ALBANY_VERBOSE
  std::cout << "in Surface Gradient Jump" << std::endl;
  std::cout << " numPlaneNodes: " << numPlaneNodes << std::endl;
  std::cout << " numPlaneDims: " << numPlaneDims << std::endl;
  std::cout << " numQPs: " << numQPs << std::endl;
  std::cout << " cubature->getNumPoints(): " << cubature->getNumPoints()
            << std::endl;
  std::cout << " cubature->getDimension(): " << cubature->getDimension()
            << std::endl;
#endif
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceScalarGradient<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(refDualBasis, fm);
  this->utils.setFieldData(refNormal, fm);
  this->utils.setFieldData(jump, fm);
  this->utils.setFieldData(nodalScalar, fm);
  this->utils.setFieldData(scalarGrad, fm);

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
SurfaceScalarGradient<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  ScalarT midPlaneAvg;
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < numQPs; ++pt) {
      minitensor::Vector<MeshScalarT> G_0(
          minitensor::Source::ARRAY, 3, refDualBasis, cell, pt, 0, 0);

      minitensor::Vector<MeshScalarT> G_1(
          minitensor::Source::ARRAY, 3, refDualBasis, cell, pt, 1, 0);

      minitensor::Vector<MeshScalarT> G_2(
          minitensor::Source::ARRAY, 3, refDualBasis, cell, pt, 2, 0);

      minitensor::Vector<MeshScalarT> N(
          minitensor::Source::ARRAY, 3, refNormal, cell, pt, 0);

      minitensor::Vector<ScalarT> scalarGradPerpendicular(0, 0, 0);
      minitensor::Vector<ScalarT> scalarGradParallel(0, 0, 0);

      // Need to inverse basis [G_0 ; G_1; G_2] and none of them should be
      // normalized
      minitensor::Tensor<MeshScalarT> gBasis(
          minitensor::Source::ARRAY, 3, refDualBasis, cell, pt, 0, 0);

      minitensor::Tensor<MeshScalarT> invRefDualBasis(3);

      // This map the position vector from parent to current configuration in
      // R^3
      gBasis          = minitensor::transpose(gBasis);
      invRefDualBasis = minitensor::inverse(gBasis);

      minitensor::Vector<MeshScalarT> invG_0(3, &invRefDualBasis(0, 0));
      minitensor::Vector<MeshScalarT> invG_1(3, &invRefDualBasis(1, 0));
      minitensor::Vector<MeshScalarT> invG_2(3, &invRefDualBasis(2, 0));

      // in-plane (parallel) contribution
      for (int node(0); node < numPlaneNodes; ++node) {
        int topNode = node + numPlaneNodes;
        midPlaneAvg =
            0.5 * (nodalScalar(cell, node) + nodalScalar(cell, topNode));
        for (int i(0); i < numDims; ++i) {
          scalarGradParallel(i) +=
              refGrads(node, pt, 0) * midPlaneAvg * invG_0(i) +
              refGrads(node, pt, 1) * midPlaneAvg * invG_1(i);
        }
      }

      // normal (perpendicular) contribution
      for (int i(0); i < numDims; ++i) {
        scalarGradPerpendicular(i) = jump(cell, pt) / thickness * invG_2(i);
      }

      // assign components to MDfield ScalarGrad
      for (int i(0); i < numDims; ++i)
        scalarGrad(cell, pt, i) =
            scalarGradParallel(i) + scalarGradPerpendicular(i);
    }
  }
}
//**********************************************************************
}  // namespace LCM
