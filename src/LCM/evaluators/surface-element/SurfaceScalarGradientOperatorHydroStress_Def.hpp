//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
SurfaceScalarGradientOperatorHydroStress<EvalT, Traits>::SurfaceScalarGradientOperatorHydroStress(
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
      val_node(p.get<std::string>("Nodal Scalar Name"), dl->node_scalar),
      surface_Grad_BF(
          p.get<std::string>("Surface Scalar Gradient Operator HydroStress Name"),
          dl->node_qp_gradient),
      grad_val_qp(
          p.get<std::string>("Surface Scalar Gradient Name"),
          dl->qp_gradient)
{
  this->addDependentField(refDualBasis);
  this->addDependentField(refNormal);
  this->addDependentField(val_node);

  // Output fields
  this->addEvaluatedField(surface_Grad_BF);
  this->addEvaluatedField(grad_val_qp);

  this->setName(
      "Surface Scalar Gradient Operator HydroStress" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_gradient->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numDims     = dims[3];

  numQPs = cubature->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims  = numDims - 1;

#ifdef ALBANY_VERBOSE
  std::cout << "in Surface Scalar Gradient Operator HydroStress" << std::endl;
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
SurfaceScalarGradientOperatorHydroStress<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(refDualBasis, fm);
  this->utils.setFieldData(refNormal, fm);
  this->utils.setFieldData(val_node, fm);
  this->utils.setFieldData(surface_Grad_BF, fm);
  this->utils.setFieldData(grad_val_qp, fm);

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
SurfaceScalarGradientOperatorHydroStress<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  minitensor::Vector<MeshScalarT> Parent_Grad_plus(3);
  minitensor::Vector<MeshScalarT> Parent_Grad_minor(3);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < numQPs; ++pt) {
      minitensor::Tensor<MeshScalarT> gBasis(
          minitensor::Source::ARRAY, 3, refDualBasis, cell, pt, 0, 0);

      minitensor::Vector<MeshScalarT> N(
          minitensor::Source::ARRAY, 3, refNormal, cell, pt, 0);

      gBasis = minitensor::transpose(gBasis);

      // in-plane (parallel) contribution
      for (int node(0); node < numPlaneNodes; ++node) {
        int topNode = node + numPlaneNodes;

        // the parallel-to-the-plane term
        for (int i(0); i < numPlaneDims; ++i) {
          Parent_Grad_plus(i)  = 0.5 * refGrads(node, pt, i);
          Parent_Grad_minor(i) = 0.5 * refGrads(node, pt, i);
        }

        // the orthogonal-to-the-plane term
        MeshScalarT invh                = 1. / thickness;
        Parent_Grad_plus(numPlaneDims)  = invh * refValues(node, pt);
        Parent_Grad_minor(numPlaneDims) = -invh * refValues(node, pt);

        // Mapping from parent to the physical domain
        minitensor::Vector<MeshScalarT> Transformed_Grad_plus(
            minitensor::dot(gBasis, Parent_Grad_plus));
        minitensor::Vector<MeshScalarT> Transformed_Grad_minor(
            minitensor::dot(gBasis, Parent_Grad_minor));

        // assign components to MDfield ScalarGrad
        for (int j(0); j < numDims; ++j) {
          surface_Grad_BF(cell, topNode, pt, j) = Transformed_Grad_plus(j);
          surface_Grad_BF(cell, node, pt, j)    = Transformed_Grad_minor(j);
        }
      }
    }
  }

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < numQPs; ++pt) {
      for (int k(0); k < numDims; ++k) {
        grad_val_qp(cell, pt, k) = 0;
        for (int node(0); node < numNodes; ++node) {
          grad_val_qp(cell, pt, k) +=
              surface_Grad_BF(cell, node, pt, k) * val_node(cell, node);
        }
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
